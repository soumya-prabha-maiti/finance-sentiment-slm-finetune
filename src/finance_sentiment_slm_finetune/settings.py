"""Environment-driven settings for the training and web app entry points."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from finance_sentiment_slm_finetune.models import MODELS, HubModel, adapter_name


class TrainSettings(BaseSettings):
    """Training run configuration; every field can be overridden by its upper-case env var."""

    model_config = SettingsConfigDict(frozen=True)

    project_dir: Path
    selected_model: str = "smollm2-360M"
    dataset_name: str = "zeroshot/twitter-financial-news-sentiment"
    max_seq_len: int = Field(default=256, gt=0)
    max_steps: int = 300
    num_epochs: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=8, gt=0)
    grad_accum: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_steps: int = Field(default=20, ge=0)
    eval_samples: int = Field(default=300, gt=0)
    eval_batch_size: int = Field(default=16, gt=0)
    max_new_tokens: int = Field(default=8, gt=0)
    seed: int = 3407
    adapter_suffix: str | None = None

    @property
    def base_model(self) -> HubModel:
        return MODELS[self.selected_model]

    @property
    def run_name(self) -> str:
        suffix: str = self.adapter_suffix or (
            f"finsent-{self.max_steps}step"
            if self.max_steps > 0
            else f"finsent-{self.num_epochs:g}ep"
        )
        return adapter_name(self.base_model.name, suffix)

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output" / self.run_name

    @property
    def adapter_dir(self) -> Path:
        return self.project_dir / "adapters" / self.run_name

    @property
    def merged_dir(self) -> Path:
        return self.project_dir / "merged_models" / self.run_name


class AppSettings(BaseSettings):
    """Web app configuration; every field can be overridden by its APP_-prefixed env var."""

    model_config = SettingsConfigDict(env_prefix="APP_", frozen=True)

    models_dir: Path | None = None
    model_name: str
    max_new_tokens: int = Field(default=8, gt=0)

    @property
    def model_source(self) -> str:
        """Directory under models_dir when one is set, otherwise a hub repo id."""
        return str(self.models_dir / self.model_name) if self.models_dir else self.model_name
