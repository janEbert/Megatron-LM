# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Any, Dict, OrderedDict

import torch
from torch import Tensor

from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import get_attention_mask


class EncoderDecoderTextGenerationController(TextGenerationController):
    """The text generation controller for encoder-decoder architecture

    This class inherits from TextGenerationController, adding features
    relating to encoder input encoder_prompt

    """

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[str, InferenceRequest],
        use_attention_mask: bool = False,
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.

        Returns:
            A dict of the inference input for the current batch.
        """
        encoder_prompts = list(
            map(lambda request: request.encoder_prompt, active_requests.values())
        )

        inference_input = self.inference_wrapped_model.prep_inference_input(
            prompts_tokens, encoder_prompts, tokenizer=self.tokenizer
        )

        if use_attention_mask and (
            attention_mask := inference_input.get("attention_mask", None) is None
        ):
            inference_input["attention_mask"] = get_attention_mask(prompts_tokens.size(1))

        return inference_input

    def _dynamic_step_context_init(
        self, construct_graph_dimensions=None, is_dummy_forward: bool = False
    ):
        """Initialize context for dynamic batching, handling encoder prefill.

        This override detects which requests need encoder prefill and prepares
        the context accordingly.

        Args:
            construct_graph_dimensions: The graph config for CUDA graphs
            is_dummy_forward (bool): Whether this is an expert parallel dummy forward

        Returns:
            input_ids (Tensor): The active input IDs
            position_ids (Tensor): The active position IDs
        """
        # Call parent implementation
        input_ids, position_ids = super()._dynamic_step_context_init(
            construct_graph_dimensions, is_dummy_forward
        )

        context = self.inference_wrapped_model.inference_context

        # Check if any requests need encoder prefill
        if context.is_encoder_decoder:
            active_request_slice = slice(
                context.paused_request_count, context.total_request_count
            )
            encoder_prefill_mask = context.get_encoder_prefill_pending_mask()

            # Store the mask for use in forward pass
            self._encoder_prefill_pending = encoder_prefill_mask[active_request_slice]
        else:
            self._encoder_prefill_pending = None

        return input_ids, position_ids

    def _dynamic_step_forward_logits(self, input_ids: Tensor, position_ids: Tensor) -> Tensor:
        """Forward step with two-phase encoder-decoder logic.

        This override implements:
        1. Encoder phase: Run encoder for requests needing prefill
        2. Decoder phase: Run decoder with cached encoder hidden states

        Args:
            input_ids (Tensor): The input token IDs
            position_ids (Tensor): The position IDs

        Returns:
            Tensor: The output logits
        """
        context = self.inference_wrapped_model.inference_context

        # Non-encoder-decoder path: use parent implementation
        if not context.is_encoder_decoder:
            return super()._dynamic_step_forward_logits(input_ids, position_ids)

        active_request_count = context.total_request_count - context.paused_request_count
        active_request_slice = slice(context.paused_request_count, context.total_request_count)

        with torch.inference_mode():
            # Phase 1: Encoder prefill for pending requests
            if self._encoder_prefill_pending is not None and self._encoder_prefill_pending.any():
                # Get the request indexes that need encoder prefill
                pending_indexes = (
                    torch.arange(
                        context.paused_request_count,
                        context.total_request_count,
                        device=input_ids.device,
                    )[self._encoder_prefill_pending]
                )

                # Build encoder input for pending requests
                encoder_tokens_list = []
                encoder_mask_list = []
                max_encoder_len = 0

                for idx in pending_indexes:
                    req_id = context.request_ids[idx].item()
                    encoder_tokens = context.pop_encoder_prompt_tokens(req_id)
                    if encoder_tokens is not None:
                        encoder_tokens_list.append(encoder_tokens)
                        max_encoder_len = max(max_encoder_len, len(encoder_tokens))
                        # Create mask for encoder (False = valid token)
                        encoder_mask_list.append(
                            torch.zeros(len(encoder_tokens), dtype=torch.bool)
                        )

                if encoder_tokens_list:
                    # Pad encoder tokens to uniform length
                    padded_encoder_tokens = []
                    padded_encoder_masks = []
                    pad_token_id = self.tokenizer.pad if hasattr(self.tokenizer, 'pad') else 0

                    for tokens, mask in zip(encoder_tokens_list, encoder_mask_list):
                        pad_len = max_encoder_len - len(tokens)
                        padded_tokens = torch.cat(
                            [tokens, torch.full((pad_len,), pad_token_id, device=tokens.device)]
                        )
                        padded_mask = torch.cat(
                            [mask, torch.ones(pad_len, dtype=torch.bool, device=mask.device)]
                        )
                        padded_encoder_tokens.append(padded_tokens)
                        padded_encoder_masks.append(padded_mask)

                    batch_encoder_tokens = torch.stack(padded_encoder_tokens)
                    batch_encoder_mask = torch.stack(padded_encoder_masks)

                    # Create dummy decoder input for encoder-only phase
                    dummy_decoder_tokens = input_ids[self._encoder_prefill_pending]
                    dummy_decoder_mask = torch.zeros_like(dummy_decoder_tokens, dtype=torch.bool)

                    # Configure attention masks
                    use_local = getattr(self.inference_wrapped_model, 'use_local', False)
                    [encoder_mask_configured, decoder_mask_configured, encoder_decoder_mask] = (
                        T5MaskedWordPieceDataset.config_attention_mask(
                            batch_encoder_tokens,
                            dummy_decoder_tokens,
                            batch_encoder_mask,
                            dummy_decoder_mask,
                            use_local,
                        )
                    )

                    # Run encoder-only forward
                    encoder_hidden_states = self.inference_wrapped_model.run_one_forward_step(
                        {
                            "encoder_tokens": batch_encoder_tokens,
                            "decoder_tokens": dummy_decoder_tokens,
                            "encoder_mask": encoder_mask_configured,
                            "decoder_mask": decoder_mask_configured,
                            "encoder_decoder_mask": encoder_decoder_mask,
                            "phase": "encoder_only",
                        }
                    )

                    # Cache encoder hidden states
                    for i, req_idx in enumerate(pending_indexes):
                        actual_seq_len = (~batch_encoder_mask[i]).sum().item()
                        context.set_encoder_hidden_states(
                            req_idx.item(),
                            encoder_hidden_states[i, :actual_seq_len],
                            actual_seq_len,
                        )

            # Phase 2: Decoder forward with cached encoder hidden states
            # Get encoder hidden states for all active requests
            encoder_hidden_states, encoder_seq_lengths = (
                context.get_encoder_hidden_states_for_batch()
            )

            # Build decoder input
            decoder_tokens = input_ids
            decoder_query_len = decoder_tokens.size(1)

            # Build encoder tokens for masking purposes (use actual encoder tokens or dummy)
            max_encoder_seq_len = encoder_seq_lengths.max().item()
            batch_encoder_tokens = torch.zeros(
                active_request_count, max_encoder_seq_len, dtype=torch.long, device=input_ids.device
            )
            batch_encoder_mask = torch.ones(
                active_request_count, max_encoder_seq_len, dtype=torch.bool, device=input_ids.device
            )

            # Set masks based on actual encoder sequence lengths
            for i, seq_len in enumerate(encoder_seq_lengths):
                if seq_len > 0:
                    batch_encoder_mask[i, :seq_len] = False

            # Decoder mask (all valid for now - context handles position properly)
            batch_decoder_mask = torch.zeros(
                active_request_count, decoder_query_len, dtype=torch.bool, device=input_ids.device
            )

            # Configure attention masks
            use_local = getattr(self.inference_wrapped_model, 'use_local', False)
            [encoder_mask_configured, decoder_mask_configured, encoder_decoder_mask] = (
                T5MaskedWordPieceDataset.config_attention_mask(
                    batch_encoder_tokens,
                    decoder_tokens,
                    batch_encoder_mask,
                    batch_decoder_mask,
                    use_local,
                )
            )

            # Run decoder forward with cached encoder hidden states
            logits = self.inference_wrapped_model.run_one_forward_step(
                {
                    "encoder_tokens": batch_encoder_tokens,
                    "decoder_tokens": decoder_tokens,
                    "encoder_mask": encoder_mask_configured,
                    "decoder_mask": decoder_mask_configured,
                    "encoder_decoder_mask": encoder_decoder_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "phase": "decoder_with_cached_encoder",
                }
            )

        # Handle pipeline parallelism broadcasting
        if self.model_is_pipeline_parallel:
            logits_seq_len = (
                active_request_count
                if context.config.materialize_only_last_token_logits
                else input_ids.size(1)
            )
            torch.distributed.broadcast(
                logits,
                self.model_pipeline_src_stage,
                group=self.model_pipeline_group,
                async_op=False,
            )
            logits = logits[:active_request_count, :logits_seq_len, :]

        return logits
