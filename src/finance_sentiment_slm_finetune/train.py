"""LoRA fine-tune a small instruct LM for 3-class financial sentiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from finance_sentiment_slm_finetune.device import resolve_device
from finance_sentiment_slm_finetune.inference import generate_replies
from finance_sentiment_slm_finetune.prompts import LABELS, build_prompt, parse_label
from finance_sentiment_slm_finetune.settings import TrainSettings

DEMO_HEADLINES: list[str] = [
    "Tech Corp announces a massive 45% plunge in quarterly revenue as chip supply lines freeze up.",
    "Global logistics market remains completely stagnant with zero growth observed throughout Q3.",
    "Renewable energy shares surge following the breakthrough approval of new green grid subsidies.",
]
LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def encode_example(
    tokenizer: PreTrainedTokenizerBase, text: str, label: str, max_seq_len: int
) -> dict[str, list[int]]:
    """Tokenizes prompt + answer with the prompt masked out of the loss.

    Args:
        tokenizer: Tokenizer carrying the model's chat template.
        text: Headline or tweet.
        label: Gold label word.
        max_seq_len: Truncation length.

    Returns:
        input_ids and labels of equal length.
    """
    prompt_ids: list[int] = tokenizer(
        build_prompt(tokenizer, text), add_special_tokens=False
    )["input_ids"]
    answer_ids: list[int] = tokenizer(
        label + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    return {
        "input_ids": (prompt_ids + answer_ids)[:max_seq_len],
        "labels": ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len],
    }


def build_training_set(
    tokenizer: PreTrainedTokenizerBase, rows: Dataset, max_seq_len: int
) -> Dataset:
    return rows.map(
        lambda row: encode_example(
            tokenizer, row["text"], LABELS[row["label"]], max_seq_len
        ),
        remove_columns=rows.column_names,
        desc="encoding",
    )


@dataclass
class PadCollator:
    """Right-pads a batch and masks padding out of the loss."""

    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width: int = max(len(f["input_ids"]) for f in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for feature in features:
            gap: int = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * gap)
            labels.append(feature["labels"] + [-100] * gap)
            attention.append([1] * len(feature["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: torch.device,
    settings: TrainSettings,
) -> list[str | None]:
    replies: list[str] = generate_replies(
        model,
        tokenizer,
        texts,
        device,
        batch_size=settings.eval_batch_size,
        max_new_tokens=settings.max_new_tokens,
    )
    return [parse_label(reply) for reply in replies]


def score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict[str, Any]],
    device: torch.device,
    settings: TrainSettings,
) -> tuple[float, int]:
    """Returns (accuracy, count of replies that named no label)."""
    model.eval()
    predictions: list[str | None] = predict(
        model, tokenizer, [row["text"] for row in rows], device, settings
    )
    correct: int = sum(
        prediction == LABELS[row["label"]] for row, prediction in zip(rows, predictions)
    )
    unparsed: int = sum(prediction is None for prediction in predictions)
    return correct / len(rows), unparsed


def main() -> None:
    settings = TrainSettings()
    base = settings.base_model
    torch.manual_seed(settings.seed)
    device: torch.device = resolve_device()
    print(f"device: {device}, model: {base.name}, run: {settings.run_name}")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        base.name, revision=base.sha
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        base.name, revision=base.sha
    ).to(device)
    print(f"base revision: {model.config._commit_hash}")

    raw = load_dataset(settings.dataset_name)
    eval_rows: list[dict[str, Any]] = list(
        raw["validation"]
        .shuffle(seed=settings.seed)
        .select(range(min(settings.eval_samples, len(raw["validation"]))))
    )
    print(f"train: {len(raw['train'])} rows, scoring on {len(eval_rows)} held-out rows")

    baseline_accuracy, baseline_unparsed = score(
        model, tokenizer, eval_rows, device, settings
    )
    print(
        f"before training: accuracy {baseline_accuracy:.1%}, "
        f"{baseline_unparsed} unparseable replies"
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()

    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(settings.output_dir),
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.grad_accum,
            max_steps=settings.max_steps,
            num_train_epochs=settings.num_epochs,
            learning_rate=settings.learning_rate,
            warmup_steps=settings.warmup_steps,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            optim="adamw_torch",
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            dataloader_pin_memory=False,
            seed=settings.seed,
            report_to=[],
        ),
        train_dataset=build_training_set(tokenizer, raw["train"], settings.max_seq_len),
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()
    model.config.use_cache = True

    tuned_accuracy, tuned_unparsed = score(model, tokenizer, eval_rows, device, settings)
    print(
        f"after training: accuracy {tuned_accuracy:.1%}, "
        f"{tuned_unparsed} unparseable replies "
        f"(delta {tuned_accuracy - baseline_accuracy:+.1%})"
    )

    for headline, prediction in zip(
        DEMO_HEADLINES, predict(model, tokenizer, DEMO_HEADLINES, device, settings)
    ):
        print(f"  {prediction or '<unparsed>'} <- {headline}")

    model.save_pretrained(settings.adapter_dir)
    print(f"saved adapter to {settings.adapter_dir}")

    merged: PreTrainedModel = model.merge_and_unload()
    merged.save_pretrained(settings.merged_dir)
    tokenizer.save_pretrained(settings.merged_dir)
    print(f"saved merged weights to {settings.merged_dir}")


if __name__ == "__main__":
    main()
