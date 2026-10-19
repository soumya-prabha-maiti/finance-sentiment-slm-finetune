"""Tests for settings, naming, and prompt helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from transformers import AutoTokenizer

from finance_sentiment_slm_finetune.models import MODELS, adapter_name
from finance_sentiment_slm_finetune.prompts import SYSTEM_PROMPT, build_prompt, parse_label
from finance_sentiment_slm_finetune.settings import AppSettings, TrainSettings

RUN_PREFIX: str = "HuggingFaceTB__SmolLM2-360M-Instruct__"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PROJECT_DIR",
        "APP_MODELS_DIR",
        "APP_MODEL_NAME",
        "MAX_STEPS",
        "NUM_EPOCHS",
        "ADAPTER_SUFFIX",
    ):
        monkeypatch.delenv(name, raising=False)


def test_adapter_name_splits_hub_id() -> None:
    assert adapter_name("org/model-1", "x") == "org__model-1__x"


def test_adapter_name_rejects_non_hub_id() -> None:
    with pytest.raises(ValueError):
        adapter_name("/local/path/model", "x")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("negative", "negative"),
        ("  Positive.", "positive"),
        ("neutral, not negative", "neutral"),
        ("I cannot tell", None),
    ],
)
def test_parse_label(reply: str, expected: str | None) -> None:
    assert parse_label(reply) == expected


def test_train_settings_requires_project_dir() -> None:
    with pytest.raises(ValidationError):
        TrainSettings()


def test_train_settings_paths_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    settings = TrainSettings()
    assert settings.run_name == f"{RUN_PREFIX}finsent-300step"
    assert settings.adapter_dir == tmp_path / "adapters" / settings.run_name
    assert settings.merged_dir == tmp_path / "merged_models" / settings.run_name
    assert settings.output_dir == tmp_path / "output" / settings.run_name


def test_train_settings_epoch_mode_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_STEPS", "-1")
    monkeypatch.setenv("NUM_EPOCHS", "2")
    assert TrainSettings().run_name == f"{RUN_PREFIX}finsent-2ep"


def test_train_settings_custom_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("ADAPTER_SUFFIX", "finsent-classweighted")
    assert TrainSettings().run_name == f"{RUN_PREFIX}finsent-classweighted"


def test_train_settings_rejects_bad_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        TrainSettings()


def test_app_settings_requires_model_name() -> None:
    with pytest.raises(ValidationError):
        AppSettings()


def test_app_settings_model_source_is_hub_id_without_models_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_MODEL_NAME", "org/model")
    assert AppSettings().model_source == "org/model"


def test_app_settings_model_source_joins_models_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APP_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("APP_MODEL_NAME", "run-a")
    assert AppSettings().model_source == str(tmp_path / "run-a")


def test_build_prompt_renders_chatml_generation_cue() -> None:
    base = MODELS["smollm2-360M"]
    tokenizer = AutoTokenizer.from_pretrained(base.name, revision=base.sha)
    prompt: str = build_prompt(tokenizer, "Apple beats estimates")
    assert SYSTEM_PROMPT in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
