"""Gradio UI for classifying financial sentiment with a merged fine-tuned model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import gradio as gr
import spaces
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from finance_sentiment_slm_finetune.device import resolve_device
from finance_sentiment_slm_finetune.inference import generate_replies
from finance_sentiment_slm_finetune.prompts import parse_label
from finance_sentiment_slm_finetune.settings import AppSettings

LOCALHOST: Final[str] = "127.0.0.1"
ALL_INTERFACES: Final[str] = "0.0.0.0"  # noqa: S104
GPU_DURATION: Final[int] = 20

EXAMPLES: list[str] = [
    "Apple beats estimates",
    "Fed holds rates steady",
    "Oil prices crash 12% as demand forecasts collapse",
]


@dataclass(frozen=True)
class LoadedModel:
    """Everything a classify call needs, resolved once at startup."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    max_new_tokens: int


# Read through module state rather than arguments: ZeroGPU pickles the arguments of a
# decorated call but inherits module globals through fork.
_loaded: LoadedModel | None = None


@spaces.GPU(duration=GPU_DURATION)
def classify(text: str) -> tuple[str, str]:
    """Classifies one headline's sentiment.

    Args:
        text: Headline to classify.

    Returns:
        The parsed label and the raw model reply.
    """
    if _loaded is None:
        raise RuntimeError("build_demo must run before classify")
    if not text.strip():
        return "", ""
    reply: str = generate_replies(
        _loaded.model,
        _loaded.tokenizer,
        [text],
        _loaded.device,
        batch_size=1,
        max_new_tokens=_loaded.max_new_tokens,
    )[0]
    return parse_label(reply) or "<unparsed>", reply


def build_demo(settings: AppSettings) -> gr.Interface:
    """Loads the model and wires it into a Gradio interface.

    Args:
        settings: App configuration.

    Returns:
        The unlaunched interface.
    """
    global _loaded
    device: torch.device = resolve_device()
    source: str = settings.model_source
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(source)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(source).to(device).eval()
    _loaded = LoadedModel(model, tokenizer, device, settings.max_new_tokens)

    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(label="Headline", lines=3),
        outputs=[gr.Textbox(label="Sentiment"), gr.Textbox(label="Raw reply")],
        examples=EXAMPLES,
        title="Financial sentiment",
        description=f"Model: {settings.model_name} on {device}",
        flagging_mode="never",
    )


def main(server_name: str = LOCALHOST) -> None:
    """Launches the app.

    Args:
        server_name: Interface address to bind.
    """
    build_demo(AppSettings()).launch(server_name=server_name)


if __name__ == "__main__":
    main()
