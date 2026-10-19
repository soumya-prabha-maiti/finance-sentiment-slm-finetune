"""Batched greedy generation shared by training eval and the web app."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from finance_sentiment_slm_finetune.prompts import build_prompt


@torch.no_grad()
def generate_replies(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Generates the model's reply to each text.

    Args:
        model: Model to generate with.
        tokenizer: Tokenizer matching the model.
        texts: Headlines or tweets to classify.
        device: Device the model lives on.
        batch_size: Texts per generate call.
        max_new_tokens: Generation length cap.

    Returns:
        The decoded replies, prompt excluded, in input order.
    """
    previous_side: str = tokenizer.padding_side
    tokenizer.padding_side = "left"
    replies: list[str] = []
    try:
        for start in range(0, len(texts), batch_size):
            chunk: list[str] = texts[start : start + batch_size]
            batch = tokenizer(
                [build_prompt(tokenizer, text) for text in chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            generated: torch.Tensor = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            replies.extend(
                tokenizer.batch_decode(
                    generated[:, batch["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
            )
    finally:
        tokenizer.padding_side = previous_side
    return replies
