"""Hyena operator as introduced by https://arxiv.org/abs/2302.10866."""

import math

try:
    from einops import rearrange
except ImportError:
    rearrange = None
import torch

from megatron import core, get_args
from megatron.core import mpu, tensor_parallel
from megatron.model.enums import AttnMaskType
from .module import MegatronModule


# def _args_to_kwargs():
#     args = get_args()

#     common_kwargs = {
#         "params_dtype": args.params_dtype,
#         "use_cpu_initialization": args.use_cpu_initialization,
#         "perform_initialization": args.perform_initialization,
#         # "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
#         # "sequence_parallel_enabled": args.sequence_parallel,
#     }
#     return common_kwargs


def initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""
    tensor_parallel.set_tensor_model_parallel_attributes(tensor=weight,
                                                         is_parallel=True,
                                                         dim=partition_dim,
                                                         stride=stride)

    with tensor_parallel.get_cuda_rng_tracker().fork():
        init_method(weight)


def get_activation_from_str(act_str):
    if act_str.lower() == "relu":
        return torch.nn.ReLU()
    elif act_str.lower() == "gelu":
        return torch.nn.GELU()
    elif act_str.lower() == "silu":
        return torch.nn.SiLU()
    elif act_str.lower() == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Activation {act_str} not supported.")


def fftconv(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = torch.nn.functional.gelu(out)

    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class Sin(torch.nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        args = get_args()
        freq = w * torch.ones(
            1,
            dim,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        )
        self.freq = torch.nn.Parameter(freq) if train_freq else freq

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()
        args = get_args()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        self.register_buffer(
            "t",
            # 1, L, 1
            torch.linspace(
                0,
                1,
                self.seq_len,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype,
            )[None, :, None],
        )

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(
            0,
            seq_len - 1,
            seq_len,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        )[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(
            1e-4,
            bands - 1,
            bands,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        )[None, None]
        z = torch.exp(-1j * f * w)
        #self.z = nn.Parameter(torch.cat([self.t, z.real, z.imag], dim=-1))
        # fix to non-learnable
        z = torch.cat([self.t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class RandomFourierPositionalEmbedding(torch.nn.Module):
    def __init__(
            self,
            emb_dim: int,
            seq_len: int,
            omega_0: float,
            use_bias: bool = False,
            **kwargs,
    ):
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even. Current {emb_dim}")
        super().__init__()
        args = get_args()

        linear_out_channels = emb_dim // 2
        self.linear = torch.nn.Linear(
            in_features=1, out_features=linear_out_channels, bias=use_bias
        )
        # initialize with xavier normal rescaled by 0.02
        torch.nn.init.xavier_normal_(self.linear.weight, gain=0.02)

        # Initialize:
        self.linear.weight.data.normal_(0.0, 2 * torch.pi * omega_0)
        if use_bias:
            torch.nn.init.constant_(self.linear.bias, 0.0)

        t = torch.linspace(
            -1,
            1,
            seq_len,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        )[None, :, None]
        self.register_buffer("t", t)

    def forward(self, L):
        out = self.linear(self.t[:, :L])
        return torch.cat([torch.cos(out), torch.sin(out)], dim=-1), (self.t + 1) / 2


class ParallelExponentialModulation(torch.nn.Module):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        args = get_args()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct

        rank = mpu.get_tensor_model_parallel_rank()
        world_size = mpu.get_tensor_model_parallel_world_size()
        partition_dim = 2

        hidden_size_per_partition = core.utils.divide(d_model, world_size)

        self.weight = torch.nn.Parameter(torch.empty(
            1,
            1,
            hidden_size_per_partition,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        ))

        master_weight = torch.linspace(
            min_decay,
            max_decay,
            d_model,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        )[None, None]
        weight_list = torch.split(
            master_weight, hidden_size_per_partition, dim=-1)
        local_weight_list = weight_list[rank::world_size]

        with torch.no_grad():
            torch.cat(local_weight_list, dim=partition_dim, out=self.weight)

    def forward(self, t, x):
        decay = torch.exp(-t * self.weight.abs())
        x = x * (decay + self.shift)
        return x


class ParallelHyenaFilter(torch.nn.Module):
    def __init__(
            self,
            config,
            d_model,
            # dim of input to MLP, augments with positional encoding
            emb_dim=3,
            order=16,  # width of the implicit MLP
            seq_len=1024,
            w=1,  # frequency of periodic activations
            omega_0=1,  # frequency of positional embeddings
            num_inner_mlps=2,
            modulate: bool = True,
            normalized=False,
            **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding
                (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        args = get_args()
        assert not args.sequence_parallel, \
            'Hyena does not support sequence parallelism'
        self.d_model = d_model
        self.modulate = modulate

        self.act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, (
            "emb_dim must be odd and greater or equal to 3 "
            "(time, sine and cosine)")
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)

        self.implicit_filter = torch.nn.Sequential(
            torch.nn.Linear(
                emb_dim,
                order,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype,
            ),
            self.act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(torch.nn.Linear(
                order,
                order,
                device=torch.cuda.current_device(),
                dtype=args.params_dtype,
            ))
            self.implicit_filter.append(self.act)

        self.final_filter = tensor_parallel.ColumnParallelLinear(
            order,
            d_model,
            config=config,
            bias=False,
            gather_output=False,
            init_method=config.init_method,
        )

        self.modulation = ParallelExponentialModulation(d_model, **kwargs)

        self.normalized = normalized

    def forward(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h, _ = self.final_filter(h)

        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.linalg.norm(h, dim=-1, ord=1, keepdim=True)
        h = rearrange(h, '1 L D -> D (1 L)')
        return h


class ParallelHyena(MegatronModule):
    """Parallel Hyena operator layer.

    Drop-in replacement for attention layers.

    The Hyena operator layer takes input with size [s, b, h] and returns
    output of the same size.

    Inputs: Q, K, V
    Operation: 1D Conv on each Q, K, V (independently)
    Long Conv(Q * K) * V, independently on each H
    """

    def __init__(
            self,
            config,
            attn_mask_type=AttnMaskType.padding,
            order=2,
            filter_order=64,
            **filter_args,
    ):
        assert rearrange is not None, \
            'Please install `einops` first, e.g., with `pip install einops`.'

        super(ParallelHyena, self).__init__()
        args = get_args()
        assert attn_mask_type is AttnMaskType.causal, \
            'Hyena currently only supports causal attention'

        self.attn_mask_type = attn_mask_type

        d_model = args.hidden_size
        l_max = args.seq_length
        order = args.num_attention_heads
        filter_order = args.hyena_filter_order
        self.act = get_activation_from_str(args.hyena_gating_activation)
        self.short_conv_L = args.hyena_short_conv_size

        modulation_args = dict(
            # Modulation kwargs.
            fast_decay_pct=args.hyena_fast_decay_pct,
            slow_decay_pct=args.hyena_slow_decay_pct,
            target=args.hyena_modulation_target,
            shift=args.hyena_shift,
        )
        filter_args = dict(
            emb_dim=args.hyena_emb_dim,
            w=args.hyena_activation_frequency,
            omega_0=args.hyena_pos_emb_frequency,
            num_inner_mlps=args.hyena_num_inner_mlps,
            modulate=args.hyena_modulate,
            normalized=args.hyena_normalized,
            **modulation_args,
        )

        self.d_model = d_model
        self.L = l_max
        self.num_heads = order

        world_size = mpu.get_tensor_model_parallel_world_size()
        hidden_size_per_partition = core.utils.divide(d_model, world_size)

        self.short_conv_weight = torch.nn.Parameter(torch.empty(
            3,  # Q, K, V
            hidden_size_per_partition,
            1,
            self.short_conv_L,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        ))
        self.short_conv_bias = torch.nn.Parameter(torch.empty(
            3,
            hidden_size_per_partition,
            device=torch.cuda.current_device(),
            dtype=args.params_dtype,
        ))
        initialize_affine_weight_gpu(
            self.short_conv_weight, config.init_method, partition_dim=1)

        self.filter = ParallelHyenaFilter(
            config,
            d_model // self.num_heads,
            order=filter_order,
            seq_len=self.L,
            **filter_args
        )
        self.long_conv_bias = torch.nn.Parameter(torch.empty(
            hidden_size_per_partition,
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        ))

    def forward(self, query_layer, key_layer, value_layer):
        # input sizes: [sq, b, np, hn]
        # seqlen, batch, tensor parallel, hidden size per tensor parallel
        np = query_layer.shape[-2] // value_layer.size(1)

        query = rearrange(query_layer, 'sq (b np) hn -> b (np hn) sq', np=np)
        key = rearrange(key_layer, 'sq (b np) hn -> b (np hn) sq', np=np)
        value = rearrange(value_layer, 'sq b np hn -> b (np hn) sq')

        q = torch.nn.functional.conv1d(
            query,
            self.short_conv_weight[0, :, :, :],
            self.short_conv_bias[0, :],
            stride=1, padding=self.short_conv_L - 1,
            dilation=1, groups=self.short_conv_weight.shape[1])[..., :self.L]
        k = torch.nn.functional.conv1d(
            key,
            self.short_conv_weight[1, :, :, :],
            self.short_conv_bias[1, :],
            stride=1, padding=self.short_conv_L - 1,
            dilation=1, groups=self.short_conv_weight.shape[1])[..., :self.L]
        v = torch.nn.functional.conv1d(
            value,
            self.short_conv_weight[2, :, :, :],
            self.short_conv_bias[2, :],
            stride=1, padding=self.short_conv_L - 1,
            dilation=1, groups=self.short_conv_weight.shape[1])[..., :self.L]

        filter = self.filter(self.L)
        filter = filter.repeat_interleave(self.num_heads, dim=0)
        # z = (1 - self.alpha1) * v * self.act(k) + self.alpha1 * v
        z = v * self.act(k)
        with torch.autocast("cuda"):
            z = fftconv(
                z.to(torch.float32),
                filter.to(torch.float32),
                self.long_conv_bias,
                None,
                gelu=False,
            )
            z = z.to(v.dtype)

        #z = (1 - self.alpha2) * z * self.act(q) + self.alpha2 * z
        z = z * self.act(q)

        return rearrange(z, 'b (np hn) sq -> b np sq hn', np=np)
