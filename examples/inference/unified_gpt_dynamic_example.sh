#!/bin/bash

# Example: GPT + Dynamic Inference Engine
# Usage: ./unified_gpt_dynamic_example.sh

# Set paths (MODIFY THESE)
CHECKPOINT_PATH="checkpoints/gpt-125m"
TOKENIZER_MODEL="tokenizers/tokenizer.model"

# Model configuration
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024

# Parallelism
NUM_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2

# Inference configuration
NUM_TOKENS_TO_GENERATE=50

# Dynamic batching configuration
BUFFER_SIZE_GB=20
MAX_REQUESTS=128
MAX_TOKENS=8192

# Prompts
PROMPTS=(
    "Hello, world!"
    "What is artificial intelligence?"
)

# Run inference
torchrun --nproc_per_node=${NUM_GPUS} \
    examples/inference/unified_batch_inference.py \
    --model-type gpt \
    --engine-type dynamic \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --load ${CHECKPOINT_PATH} \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --prompts "${PROMPTS[@]}" \
    --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
    --inference-dynamic-batching-buffer-size-gb ${BUFFER_SIZE_GB} \
    --inference-dynamic-batching-max-requests ${MAX_REQUESTS} \
    --inference-dynamic-batching-max-tokens ${MAX_TOKENS} \
    --position-embedding-type rope \
    --bf16 \
    --micro-batch-size 1
