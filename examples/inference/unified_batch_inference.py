#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unified batch inference script supporting:
- Model types: GPT (decoder-only) and T5 (encoder-decoder)
- Engine types: Static and Dynamic inference engines
- Hardware: Multi-GPU with tensor parallelism
- Usage: Simple command-line batch testing (no Slurm required)

IMPORTANT: T5 + Dynamic engine is NOT currently supported because T5 lacks KV cache
support (see t5_inference_wrapper.py:166,214).
"""

import json
import os
import sys
import time
from functools import partial
from typing import List, Dict, Any

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core import mpu
from megatron.core.inference.engines import StaticInferenceEngine, DynamicInferenceEngine
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.t5.t5_inference_wrapper import (
    T5InferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (
    EncoderDecoderTextGenerationController,
)
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.inference.utils import add_inference_args, get_inference_config_from_model_and_args


# ============================================================================
# Command-Line Arguments
# ============================================================================


def add_unified_inference_args(parser):
    """Add arguments for unified inference script."""
    # Add base inference args first
    parser = add_inference_args(parser)

    group = parser.add_argument_group(title='unified inference')

    # Model and engine selection
    group.add_argument(
        "--model-type",
        type=str,
        choices=["gpt", "t5"],
        default="gpt",
        help="Type of model to run inference with",
    )
    group.add_argument(
        "--engine-type",
        type=str,
        choices=["static", "dynamic"],
        default="static",
        help="Type of inference engine to use",
    )

    # T5-specific prompts
    group.add_argument(
        "--encoder-prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Encoder input prompts for T5 models',
    )

    # Static engine specific args
    group.add_argument(
        "--inference-max-requests",
        type=int,
        default=8,
        help='Maximum batch size for static inference engine',
    )
    group.add_argument(
        "--inference-max-seq-length",
        type=int,
        default=2048,
        help='Maximum sequence length for static inference engine',
    )

    return parser


# ============================================================================
# Model Providers
# ============================================================================


def gpt_model_provider(pre_process=True, post_process=True):
    """Build GPT model."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    config = core_transformer_config_from_args(args)

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
        ),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )
    return model


def t5_model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build T5 model."""
    from copy import deepcopy
    from megatron.core.models.T5 import T5Model
    from megatron.core.models.T5.t5_spec import (
        get_t5_encoder_with_transformer_engine_block_spec,
        get_t5_decoder_with_transformer_engine_block_spec,
        get_t5_encoder_with_local_block_spec,
        get_t5_decoder_with_local_block_spec,
    )
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    config = core_transformer_config_from_args(args)

    # Encoder config (may have different num_layers)
    encoder_config = deepcopy(config)
    encoder_config.num_layers = args.encoder_num_layers

    # Get layer specs
    use_te = args.transformer_impl != "local"
    if use_te:
        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(args.encoder_num_layers)
        de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(args.num_layers)
    else:
        en_block_spec = get_t5_encoder_with_local_block_spec(args.encoder_num_layers)
        de_block_spec = get_t5_decoder_with_local_block_spec(args.num_layers)

    model = T5Model(
        config=config,
        encoder_config=encoder_config,
        transformer_encoder_layer_spec=en_block_spec,
        transformer_decoder_layer_spec=de_block_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
    )
    return model


# ============================================================================
# Engine Builders
# ============================================================================


def build_static_gpt_engine(args, model, tokenizer):
    """Build StaticInferenceEngine for GPT."""
    inference_context = StaticInferenceContext(
        args.inference_max_requests, args.inference_max_seq_length
    )
    wrapped_model = GPTInferenceWrapper(model, inference_context)
    controller = TextGenerationController(inference_wrapped_model=wrapped_model, tokenizer=tokenizer)
    return StaticInferenceEngine(text_generation_controller=controller)


def build_static_t5_engine(args, model, tokenizer):
    """Build StaticInferenceEngine for T5."""
    use_local = getattr(args, 'transformer_impl', 'transformer_engine') == 'local'

    # T5 static inference doesn't use KV cache, so we pass None for context
    # The use_local flag indicates whether to use local transformer impl or TE
    wrapped_model = T5InferenceWrapper(model, inference_context=None, use_local=use_local)
    controller = EncoderDecoderTextGenerationController(
        inference_wrapped_model=wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(text_generation_controller=controller)


def build_dynamic_gpt_engine(args, model, tokenizer):
    """Build DynamicInferenceEngine for GPT."""
    inference_config = get_inference_config_from_model_and_args(model, args)
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)
    return DynamicInferenceEngine(controller, context)


def get_inference_engine(args, model, tokenizer):
    """Factory function to create appropriate inference engine."""
    model_type = args.model_type
    engine_type = args.engine_type

    # Validate combination
    if model_type == "t5" and engine_type == "dynamic":
        raise NotImplementedError(
            "T5 + Dynamic inference is NOT supported.\n"
            "T5 lacks KV cache support required for dynamic batching.\n"
            "See t5_inference_wrapper.py lines 166 and 214.\n"
            "Use --engine-type static for T5 models."
        )

    # Build engine
    if model_type == "gpt":
        if engine_type == "static":
            return build_static_gpt_engine(args, model, tokenizer)
        else:
            return build_dynamic_gpt_engine(args, model, tokenizer)
    else:  # t5
        return build_static_t5_engine(args, model, tokenizer)


# ============================================================================
# Main
# ============================================================================


@torch.inference_mode()
def main():
    """Run unified batch inference."""
    # Initialize
    initialize_megatron(
        extra_args_provider=add_unified_inference_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()
    print_rank_0(f"\n{'='*60}")
    print_rank_0(f"Unified Batch Inference")
    print_rank_0(f"Model type: {args.model_type}")
    print_rank_0(f"Engine type: {args.engine_type}")
    print_rank_0(f"{'='*60}\n")

    # Select model provider
    if args.model_type == "gpt":
        model_provider_fn = gpt_model_provider
    else:
        model_provider_fn = t5_model_provider

    # Load model
    model = get_model(model_provider_fn, wrap_with_ddp=False)
    if args.load is not None:
        load_checkpoint(model, None, None, strict=False)

    assert len(model) == 1, "Virtual pipeline parallelism is not supported"
    model = model[0]
    model.eval()

    # Build tokenizer
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Build engine
    engine = get_inference_engine(args, model, tokenizer)

    # Build sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_tokens_to_generate=args.num_tokens_to_generate,
        return_log_probs=args.return_log_probs,
    )

    # Prepare prompts
    if args.model_type == "gpt":
        prompts = args.prompts or ["Hello, I am a language model"]
        encoder_prompts = None
        print_rank_0(f"Running inference on {len(prompts)} GPT prompts")
    else:  # T5
        encoder_prompts = args.encoder_prompts or ["Translate English to French: Hello"]
        prompts = [""] * len(encoder_prompts)  # T5 decoder starts empty
        print_rank_0(f"Running inference on {len(encoder_prompts)} T5 encoder prompts")

    # Run inference
    print_rank_0(f"\n{'='*60}")
    print_rank_0(f"Generating tokens...")
    print_rank_0(f"{'='*60}\n")

    start_time = time.time()

    if args.model_type == "t5":
        results = engine.generate(
            prompts=prompts, encoder_prompts=encoder_prompts, add_BOS=True, sampling_params=sampling_params
        )
    else:
        results = engine.generate(prompts=prompts, sampling_params=sampling_params)

    elapsed = time.time() - start_time

    # Print results
    if torch.distributed.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"RESULTS (elapsed: {elapsed:.3f}s)")
        print(f"{'='*60}\n")
        for idx, result in enumerate(results):
            print(f"--- Prompt {idx} ---")
            if args.model_type == "t5":
                print(f"Encoder: {encoder_prompts[idx]}")
            print(f"Input: {result.prompt}")
            print(f"Output: {result.generated_text}")
            print(f"Tokens: {len(result.generated_tokens)}")
            print()

        # Save to file if requested
        if args.output_path:
            output_data = []
            for idx, result in enumerate(results):
                output_data.append(
                    {
                        "request_id": result.request_id,
                        "prompt": result.prompt,
                        "encoder_prompt": encoder_prompts[idx] if encoder_prompts else None,
                        "generated_text": result.generated_text,
                        "generated_tokens": result.generated_tokens,
                    }
                )
            with open(args.output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output_path}\n")

        print(f"{'='*60}")
        print(f"Summary:")
        print(f"  Model: {args.model_type}")
        print(f"  Engine: {args.engine_type}")
        print(f"  Prompts: {len(results)}")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Avg time per prompt: {elapsed/len(results):.3f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
