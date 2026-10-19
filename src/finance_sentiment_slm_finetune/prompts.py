"""Prompt format and label parsing shared by training and inference."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

LABELS: dict[int, str] = {0: "negative", 1: "positive", 2: "neutral"}
SYSTEM_PROMPT: str = (
    "You are a financial sentiment classifier. Reply with exactly one word: "
    "negative, neutral, or positive."
)


def build_prompt(tokenizer: PreTrainedTokenizerBase, text: str) -> str:
    """Renders the chat prompt up to the assistant's turn.

    Args:
        tokenizer: Tokenizer carrying the model's chat template.
        text: Headline or tweet to classify.

    Returns:
        The rendered prompt string.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def parse_label(completion: str) -> str | None:
    """Returns the earliest label named in a reply, or None if it names none.

    Args:
        completion: The model's decoded reply.

    Returns:
        The matched label.
    """
    cleaned: str = completion.strip().lower()
    hits: list[tuple[int, str]] = [
        (cleaned.find(name), name) for name in LABELS.values() if name in cleaned
    ]
    return min(hits)[1] if hits else None
