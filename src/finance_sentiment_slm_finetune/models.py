"""Registry of supported base models and adapter naming."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HubModel:
    """A Hugging Face hub model pinned to a commit."""

    name: str
    sha: str


MODELS: dict[str, HubModel] = {
    "smollm2-360M": HubModel(
        name="HuggingFaceTB/SmolLM2-360M-Instruct",
        sha="a10cc1512eabd3dde888204e902eca88bddb4951",
    ),
}


def adapter_name(model_name: str, suffix: str) -> str:
    """Builds '<org>__<model>__<suffix>' from a hub id like 'org/model'.

    Args:
        model_name: Hugging Face hub id of the base model.
        suffix: Free-form tag identifying this fine-tune.

    Returns:
        The adapter directory name.
    """
    org, model = model_name.split("/")
    return f"{org}__{model}__{suffix}"
