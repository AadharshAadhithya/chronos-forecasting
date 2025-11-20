import types
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer


def make_t5_decoder_bidirectional(model: torch.nn.Module) -> None:
    """
    Patch a T5-style seq2seq model so that the decoder self-attention is
    bidirectional instead of causal.

    This is done without modifying the transformers source code by:
    - overriding `decoder._update_causal_mask` to only encode padding (no causal triangle)
    - setting each decoder self-attention module's `is_decoder` flag to False so
      that relative position biases are computed bidirectionally.
    """

    def non_causal_update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool,
    ):
        # No special causal structure: just standard pad masking.
        if attention_mask is None:
            return None

        # (batch, seq) -> (batch, 1, 1, seq), will broadcast over query_length
        mask = attention_mask[:, None, None, :]
        mask = mask.to(dtype=input_tensor.dtype)
        mask = (1.0 - mask) * torch.finfo(input_tensor.dtype).min
        return mask

    # Only patch models that expose a T5-style decoder stack with _update_causal_mask
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "_update_causal_mask"):
        return

    # Override decoder causal mask computation
    decoder._update_causal_mask = types.MethodType(non_causal_update_causal_mask, decoder)

    # Make decoder self-attention relative position bias bidirectional
    block_list = getattr(decoder, "block", None)
    if block_list is None:
        return

    for block in block_list:
        # T5Block: layer[0] is T5LayerSelfAttention, which has .SelfAttention (T5Attention)
        layer_list = getattr(block, "layer", None)
        if not layer_list:
            continue

        self_attn = getattr(layer_list[0], "SelfAttention", None)
        if self_attn is not None and hasattr(self_attn, "is_decoder"):
            self_attn.is_decoder = False


class DiffTrainer(Trainer):
    """
    Custom Trainer for diffusion-style denoising training on discrete tokens.

    The training objective:
    - Input: clean `input_ids` of shape (b, L)
    - With 1% probability, randomly truncate the sequence length to a value
      uniformly sampled from [1, L].
    - Apply `forward_process` to sample a per-sequence masking probability and
      replace tokens with a dedicated [MASK] token.
    - Compute token-level cross-entropy only on masked positions, reweighted
      by the per-sequence masking probability.
    """

    def __init__(self, mask_token_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_token_id = mask_token_id

    @staticmethod
    def forward_process(
        input_ids: torch.Tensor,
        mask_token_id: int,
        eps: float = 1e-3,
    ):
        """
        Implements:

        def forward_process(input_ids, eps=1e-3):
            b, l = input_ids.shape
            t = torch.rand(b, device=input_ids.device)
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)
            masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
            noisy_batch = torch.where(masked_indices, MASK_ID, input_ids)
            return noisy_batch, masked_indices, p_mask
        """
        b, l = input_ids.shape

        t = torch.rand(b, device=input_ids.device)
        p_mask = (1.0 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(
            masked_indices,
            torch.full_like(input_ids, mask_token_id),
            input_ids,
        )

        return noisy_batch, masked_indices, p_mask

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        """
        Replace the standard language-modeling loss with the custom denoising loss.

        The input batch is expected to contain at least:
        - input_ids: (b, L)
        - attention_mask: (b, L) (optional but recommended)
        Any provided labels are ignored.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask")

        # With 1% probability, randomly truncate the sequence length for this batch.
        if torch.rand(1) < 0.01:
            max_len = input_ids.shape[1]
            random_length = torch.randint(
                1,
                max_len + 1,
                (1,),
                device=input_ids.device,
            )
            new_len = int(random_length.item())
            input_ids = input_ids[:, :new_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :new_len]

        noisy_batch, masked_indices, p_mask = self.forward_process(
            input_ids=input_ids,
            mask_token_id=self.mask_token_id,
        )

        outputs = model(input_ids=noisy_batch, attention_mask=attention_mask)
        logits = outputs.logits  # (b, L, vocab)

        # Move masks to the same device as logits
        masked_indices = masked_indices.to(logits.device)
        p_mask = p_mask.to(logits.device)
        input_ids = input_ids.to(logits.device)

        # Cross-entropy over masked positions only, reweighted by 1/p_mask
        token_loss = F.cross_entropy(
            logits[masked_indices],
            input_ids[masked_indices],
            reduction="none",
        ) / p_mask[masked_indices]

        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        return (loss, outputs) if return_outputs else loss


