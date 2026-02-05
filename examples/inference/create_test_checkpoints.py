#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Creates small test checkpoints for unified_batch_inference.py testing.

Usage:
    python examples/inference/create_test_checkpoints.py --output-dir ./test_checkpoints

Creates:
    ./test_checkpoints/
    ├── gpt-tiny/          # GPT checkpoint (TP=1)
    ├── gpt-tiny-tp2/      # GPT checkpoint (TP=2)
    ├── t5-tiny/           # T5 checkpoint (TP=1)
    └── tokenizers/
        └── bert-vocab.txt # BERT vocab for T5
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def download_bert_vocab(output_dir: Path):
    """Download BERT vocab file for T5 tokenizer."""
    vocab_path = output_dir / "tokenizers" / "bert-vocab.txt"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if vocab_path.exists():
        print(f"BERT vocab already exists at {vocab_path}")
        return vocab_path

    print(f"Downloading BERT vocab to {vocab_path}...")
    import urllib.request

    url = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
    try:
        urllib.request.urlretrieve(url, vocab_path)
        print(f"Successfully downloaded BERT vocab ({vocab_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"Warning: Failed to download BERT vocab: {e}")
        print("T5 checkpoint creation will be skipped")
        return None

    return vocab_path


def create_gpt_checkpoint(output_dir: Path, tp_size: int = 1, num_gpus: int = 1):
    """Create minimal GPT checkpoint using pretrain_gpt.py with mock data."""
    ckpt_name = f"gpt-tiny-tp{tp_size}" if tp_size > 1 else "gpt-tiny"
    ckpt_path = output_dir / ckpt_name

    if ckpt_path.exists():
        print(f"GPT checkpoint already exists at {ckpt_path}, skipping creation")
        return ckpt_path

    print(f"Creating GPT checkpoint (TP={tp_size}) at {ckpt_path}...")

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "pretrain_gpt.py",
        # Minimal model
        "--num-layers",
        "2",
        "--hidden-size",
        "64",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "128",
        "--max-position-embeddings",
        "128",
        # Parallelism
        "--tensor-model-parallel-size",
        str(tp_size),
        # Training
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(num_gpus),
        "--train-iters",
        "10",
        "--lr",
        "1e-4",
        "--bf16",
        # Mock data
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "1000",
        # Checkpoint
        "--save",
        str(ckpt_path),
        "--save-interval",
        "10",
        "--no-save-optim",
        "--no-save-rng",
        # Disable logging
        "--log-interval",
        "10",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Successfully created GPT checkpoint at {ckpt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating GPT checkpoint: {e}")
        raise

    return ckpt_path


def create_t5_checkpoint(output_dir: Path, vocab_path: Path, tp_size: int = 1, num_gpus: int = 1):
    """Create minimal T5 checkpoint programmatically."""
    ckpt_name = f"t5-tiny-tp{tp_size}" if tp_size > 1 else "t5-tiny"
    ckpt_path = output_dir / ckpt_name

    if ckpt_path.exists():
        print(f"T5 checkpoint already exists at {ckpt_path}, skipping creation")
        return ckpt_path

    if vocab_path is None:
        print("Skipping T5 checkpoint creation (no vocab file)")
        return None

    print(f"Creating T5 checkpoint (TP={tp_size}) at {ckpt_path}...")

    # Create a temporary Python script for T5 checkpoint creation
    script_content = f"""
import os
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.T5.t5_model import T5Model
from megatron.core.models.T5.t5_spec import (
    get_t5_encoder_with_local_block_spec,
    get_t5_decoder_with_local_block_spec,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import dist_checkpointing

def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    # Initialize parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size={tp_size},
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")

    # Create config
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
        ffn_hidden_size=256,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )

    # Create T5 model
    model = T5Model(
        config=config,
        transformer_encoder_layer_spec=get_t5_encoder_with_local_block_spec(),
        transformer_decoder_layer_spec=get_t5_decoder_with_local_block_spec(),
        vocab_size=30522,  # BERT vocab size
        max_sequence_length=128,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type='learned_absolute',
        rotary_percent=1.0,
    )

    # Move to device
    if torch.cuda.is_available():
        model = model.to(device)

    # Save checkpoint
    ckpt_path = "{ckpt_path}"
    os.makedirs(ckpt_path, exist_ok=True)

    # Create iteration directory
    iter_dir = os.path.join(ckpt_path, "iter_0000010")
    os.makedirs(iter_dir, exist_ok=True)

    # Save using dist_checkpointing
    sharded_state_dict = model.sharded_state_dict(prefix="model.")
    dist_checkpointing.save(sharded_state_dict, iter_dir)

    # Create latest_checkpointed_iteration.txt
    with open(os.path.join(ckpt_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("10")

    print(f"Successfully created T5 checkpoint at {{ckpt_path}}")

    # Cleanup
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
"""

    # Write temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Run the script with torchrun
        cmd = ["torchrun", f"--nproc_per_node={num_gpus}", temp_script]
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Successfully created T5 checkpoint at {ckpt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating T5 checkpoint: {e}")
        raise
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.unlink(temp_script)

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Create test checkpoints for unified_batch_inference.py"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--gpt-only", action="store_true", help="Create only GPT checkpoints")
    parser.add_argument("--t5-only", action="store_true", help="Create only T5 checkpoints")
    parser.add_argument(
        "--tp2", action="store_true", help="Also create TP=2 checkpoints (requires 2 GPUs)"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading BERT vocab"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test checkpoints in {output_dir}")
    print("=" * 80)

    # Download tokenizers
    vocab_path = None
    if not args.gpt_only and not args.skip_download:
        vocab_path = download_bert_vocab(output_dir)
        print()

    # Create GPT checkpoints
    if not args.t5_only:
        try:
            create_gpt_checkpoint(output_dir, tp_size=1, num_gpus=1)
            print()
        except Exception as e:
            print(f"Failed to create GPT TP=1 checkpoint: {e}")
            sys.exit(1)

        if args.tp2:
            try:
                create_gpt_checkpoint(output_dir, tp_size=2, num_gpus=2)
                print()
            except Exception as e:
                print(f"Failed to create GPT TP=2 checkpoint: {e}")
                sys.exit(1)

    # Create T5 checkpoints
    if not args.gpt_only:
        try:
            create_t5_checkpoint(output_dir, vocab_path, tp_size=1, num_gpus=1)
            print()
        except Exception as e:
            print(f"Failed to create T5 TP=1 checkpoint: {e}")
            sys.exit(1)

        if args.tp2:
            try:
                create_t5_checkpoint(output_dir, vocab_path, tp_size=2, num_gpus=2)
                print()
            except Exception as e:
                print(f"Failed to create T5 TP=2 checkpoint: {e}")
                sys.exit(1)

    print("=" * 80)
    print("Checkpoint creation complete!")
    print(f"\nCheckpoints created in: {output_dir}")
    print("\nTo test with unified_batch_inference.py:")
    print("\n  # GPT:")
    print(
        f"  torchrun --nproc_per_node=1 examples/inference/unified_batch_inference.py \\"
    )
    print(f"      --model-type gpt --engine-type static \\")
    print(f"      --tensor-model-parallel-size 1 \\")
    print(f"      --num-layers 2 --hidden-size 64 --num-attention-heads 4 \\")
    print(f"      --seq-length 128 --max-position-embeddings 128 \\")
    print(f"      --load {output_dir}/gpt-tiny \\")
    print(f"      --tokenizer-type NullTokenizer --vocab-size 1000 \\")
    print(f'      --prompts "test prompt"')
    print("\n  # T5:")
    print(
        f"  torchrun --nproc_per_node=1 examples/inference/unified_batch_inference.py \\"
    )
    print(f"      --model-type t5 --engine-type static \\")
    print(f"      --tensor-model-parallel-size 1 \\")
    print(f"      --num-layers 2 --encoder-num-layers 2 \\")
    print(f"      --hidden-size 64 --num-attention-heads 4 \\")
    print(f"      --seq-length 128 --max-position-embeddings 128 \\")
    print(f"      --load {output_dir}/t5-tiny \\")
    print(f"      --tokenizer-type BertWordPieceLowerCase \\")
    print(f"      --vocab-file {output_dir}/tokenizers/bert-vocab.txt \\")
    print(f'      --encoder-prompts "translate: hello"')


if __name__ == "__main__":
    main()
