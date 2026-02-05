# Unified Batch Inference Script

A unified script for running batch inference with Megatron-LM that supports multiple model types and inference engines.

## Features

- **Model Types**:
  - GPT (decoder-only models)
  - T5 (encoder-decoder models)

- **Engine Types**:
  - Static inference engine (batch processing)
  - Dynamic inference engine (with KV cache and dynamic batching)

- **Hardware Support**: Multi-GPU with tensor parallelism via `torchrun`

## Important Limitations

⚠️ **T5 + Dynamic Engine is NOT supported** because T5 lacks KV cache support (see `megatron/core/inference/model_inference_wrappers/t5/t5_inference_wrapper.py` lines 166 and 214). Use `--engine-type static` for T5 models.

## Usage

### Basic Command Structure

```bash
torchrun --nproc_per_node=<NUM_GPUS> examples/inference/unified_batch_inference.py \
    --model-type <gpt|t5> \
    --engine-type <static|dynamic> \
    --tensor-model-parallel-size <TP_SIZE> \
    [MODEL_ARGS] \
    [INFERENCE_ARGS]
```

### Example 1: GPT + Static Engine (2 GPUs)

```bash
torchrun --nproc_per_node=2 examples/inference/unified_batch_inference.py \
    --model-type gpt \
    --engine-type static \
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --load checkpoints/gpt-125m \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --prompts "Hello, world!" "What is AI?" "Explain machine learning." \
    --num-tokens-to-generate 50 \
    --inference-max-requests 8 \
    --inference-max-seq-length 2048
```

### Example 2: GPT + Dynamic Engine (2 GPUs)

```bash
torchrun --nproc_per_node=2 examples/inference/unified_batch_inference.py \
    --model-type gpt \
    --engine-type dynamic \
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --load checkpoints/gpt-125m \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --prompts "Hello, world!" "What is AI?" \
    --num-tokens-to-generate 50 \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --inference-dynamic-batching-max-requests 128 \
    --inference-dynamic-batching-max-tokens 8192
```

### Example 3: T5 + Static Engine (2 GPUs)

```bash
torchrun --nproc_per_node=2 examples/inference/unified_batch_inference.py \
    --model-type t5 \
    --engine-type static \
    --tensor-model-parallel-size 2 \
    --num-layers 6 \
    --encoder-num-layers 6 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load checkpoints/t5-small \
    --vocab-file /path/to/vocab.txt \
    --encoder-prompts "Translate to French: Hello" "Summarize: The quick brown fox jumps over the lazy dog." \
    --num-tokens-to-generate 30 \
    --inference-max-requests 8 \
    --inference-max-seq-length 512
```

### Example 4: Saving Results to JSON

```bash
torchrun --nproc_per_node=2 examples/inference/unified_batch_inference.py \
    --model-type gpt \
    --engine-type static \
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --load checkpoints/gpt-125m \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --prompts "Hello, world!" \
    --num-tokens-to-generate 50 \
    --output-path results.json
```

## Command-Line Arguments

### Model and Engine Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-type` | str | `gpt` | Model type: `gpt` or `t5` |
| `--engine-type` | str | `static` | Engine type: `static` or `dynamic` |

### Prompt Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompts` | str[] | `["Hello, I am a language model"]` | Decoder prompts (for GPT) |
| `--encoder-prompts` | str[] | `["Translate English to French: Hello"]` | Encoder prompts (for T5 only) |
| `--num-tokens-to-generate` | int | `30` | Number of tokens to generate per prompt |

### Static Engine Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--inference-max-requests` | int | `8` | Maximum batch size for static engine |
| `--inference-max-seq-length` | int | `2048` | Maximum sequence length for static engine |

### Dynamic Engine Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--inference-dynamic-batching-buffer-size-gb` | float | - | KV cache buffer size in GB |
| `--inference-dynamic-batching-max-requests` | int | - | Maximum concurrent requests |
| `--inference-dynamic-batching-max-tokens` | int | - | Maximum tokens in batch |

### Sampling Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | `1.0` | Sampling temperature |
| `--top_k` | int | `1` | Top-k sampling |
| `--top_p` | float | `0.0` | Top-p (nucleus) sampling |

### Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-path` | str | `None` | Path to save results as JSON |

## Model Configuration

### GPT Model Arguments

```bash
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024 \
--position-embedding-type rope \
--tokenizer-type GPTSentencePieceTokenizer \
--tokenizer-model /path/to/tokenizer.model
```

### T5 Model Arguments

```bash
--num-layers 6 \
--encoder-num-layers 6 \
--hidden-size 512 \
--num-attention-heads 8 \
--seq-length 512 \
--encoder-seq-length 512 \
--decoder-seq-length 128 \
--max-position-embeddings 512 \
--vocab-file /path/to/vocab.txt \
--tokenizer-type BertWordPieceTokenizer
```

## Obtaining Test Models

### Option 1: Train Small Models with Mock Data (Simplest)

**GPT 125M:**
```bash
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --train-iters 100 \
    --save checkpoints/gpt-125m \
    --save-interval 100 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --mock-data \
    --fp16
```

**T5 Small:**
```bash
torchrun --nproc_per_node=2 pretrain_t5.py \
    --tensor-model-parallel-size 2 \
    --num-layers 6 \
    --encoder-num-layers 6 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --train-iters 100 \
    --save checkpoints/t5-small \
    --save-interval 100 \
    --vocab-file /path/to/bert-vocab.txt \
    --mock-data \
    --fp16
```

### Option 2: Download Pre-trained Tokenizers

**For GPT (SentencePiece):**
```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-2-7b-hf', 'tokenizer.model', local_dir='./tokenizers')"
```

**For T5:**
```bash
python -c "from transformers import T5Tokenizer; t = T5Tokenizer.from_pretrained('t5-small'); t.save_pretrained('./tokenizers/t5')"
```

### Option 3: Convert HuggingFace Models

```bash
# Convert GPT-2
python tools/checkpoint/convert.py \
    --model-type GPT2 \
    --loader hf \
    --saver mcore \
    --load-dir gpt2 \
    --save-dir checkpoints/gpt2-mcore \
    --target-tensor-parallel-size 2

# Convert T5-small
python tools/checkpoint/convert.py \
    --model-type t5 \
    --loader hf \
    --saver mcore \
    --load-dir t5-small \
    --save-dir checkpoints/t5-small-mcore \
    --target-tensor-parallel-size 2
```

## Test Matrix

| Model | Engine | Expected Result |
|-------|--------|-----------------|
| GPT + Static | ✅ Should work |
| GPT + Dynamic | ✅ Should work |
| T5 + Static | ✅ Should work |
| T5 + Dynamic | ❌ NotImplementedError (expected) |

## Output Format

The script prints results to stdout with the following format:

```
============================================================
RESULTS (elapsed: 2.345s)
============================================================

--- Prompt 0 ---
Input: Hello, world!
Output: This is a generated text response...
Tokens: 50

============================================================
Summary:
  Model: gpt
  Engine: static
  Prompts: 3
  Total time: 2.345s
  Avg time per prompt: 0.782s
============================================================
```

If `--output-path` is specified, results are also saved to a JSON file:

```json
[
  {
    "request_id": 0,
    "prompt": "Hello, world!",
    "encoder_prompt": null,
    "generated_text": "This is a generated text response...",
    "generated_tokens": [1234, 5678, ...]
  }
]
```

## Future Work: T5 + Dynamic Engine Support

To enable T5 with dynamic inference engine, the following modifications are needed:

1. **Modify T5InferenceWrapper** to pass `inference_context` to the model and handle prefill vs decode modes
2. **Extend DynamicInferenceContext** to cache encoder hidden states per request
3. **Update EncoderDecoderTextGenerationController** to support async generation methods
4. **Modify T5Model.forward()** to reuse cached encoder outputs during decode steps

See the detailed implementation plan in the original plan document for code snippets and step-by-step instructions.

## Troubleshooting

### Error: "T5 + Dynamic inference is NOT supported"

This is expected. T5 lacks KV cache support required for dynamic batching. Use `--engine-type static` instead.

### Error: "request idxs with prompts longer than context.max_tokens"

Your prompts are too long for the configured context. Either:
- Reduce prompt length
- Increase `--inference-dynamic-batching-max-tokens` (dynamic engine)
- Increase `--inference-max-seq-length` (static engine)

### Out of Memory Errors

For dynamic engine:
- Reduce `--inference-dynamic-batching-buffer-size-gb`
- Reduce `--inference-dynamic-batching-max-requests`
- Reduce `--inference-dynamic-batching-max-tokens`

For static engine:
- Reduce `--inference-max-requests`
- Reduce `--inference-max-seq-length`

## References

- GPT Dynamic Inference: `examples/inference/gpt/gpt_dynamic_inference.py`
- GPT Static Inference: `examples/inference/gpt/gpt_static_inference.py`
- T5 Static Inference: `examples/inference/t5/simple_t5_batch_inference.py`
- Megatron Core Inference: `megatron/core/inference/`
