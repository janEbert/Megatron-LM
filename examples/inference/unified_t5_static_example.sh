#!/bin/bash

# Example: T5 + Static Inference Engine
# Usage: ./unified_t5_static_example.sh

# Set paths (MODIFY THESE)
CHECKPOINT_PATH="checkpoints/t5-small"
VOCAB_FILE="tokenizers/bert-vocab.txt"

# Model configuration
NUM_LAYERS=6
ENCODER_NUM_LAYERS=6
HIDDEN_SIZE=512
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=512
MAX_POSITION_EMBEDDINGS=512

# Parallelism
NUM_GPUS=2
TENSOR_MODEL_PARALLEL_SIZE=2

# Inference configuration
NUM_TOKENS_TO_GENERATE=30
INFERENCE_MAX_REQUESTS=8
INFERENCE_MAX_SEQ_LENGTH=512

# Encoder prompts (for T5)
ENCODER_PROMPTS=(
    "Translate to French: Hello, how are you?"
    "Summarize: The quick brown fox jumps over the lazy dog."
    "Question: What is the capital of France? Context: France is a country in Europe."
)

# Run inference
torchrun --nproc_per_node=${NUM_GPUS} \
    examples/inference/unified_batch_inference.py \
    --model-type t5 \
    --engine-type static \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --encoder-num-layers ${ENCODER_NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --encoder-seq-length ${SEQ_LENGTH} \
    --decoder-seq-length 128 \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --load ${CHECKPOINT_PATH} \
    --tokenizer-type BertWordPieceTokenizer \
    --vocab-file ${VOCAB_FILE} \
    --encoder-prompts "${ENCODER_PROMPTS[@]}" \
    --num-tokens-to-generate ${NUM_TOKENS_TO_GENERATE} \
    --inference-max-requests ${INFERENCE_MAX_REQUESTS} \
    --inference-max-seq-length ${INFERENCE_MAX_SEQ_LENGTH} \
    --bf16 \
    --micro-batch-size 1
