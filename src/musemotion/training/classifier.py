from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from musemotion.config import resolve_path
from musemotion.data.goemotions import label_names_from_goemotions_features
from musemotion.emotions import EMOPIA_QUADRANTS, map_goemotion_labels_to_quadrant


def train_classifier(config: dict[str, Any]) -> None:
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    model_name = config["model"].get("name", "bert-base-uncased")
    max_length = int(config["model"].get("max_length", 128))
    data_config = config.get("data", {})
    text_column = data_config.get("text_column", "text")
    label_column = data_config.get("label_column", "labels")
    dataset = load_dataset(data_config.get("dataset_name", "go_emotions"))
    label_names = label_names_from_goemotions_features(dataset["train"].features)

    def add_quadrant(example: dict[str, Any]) -> dict[str, Any]:
        original_labels = [label_names[index] for index in example[label_column]]
        quadrant = map_goemotion_labels_to_quadrant(original_labels)
        return {
            "emotion_id": -1 if quadrant is None else quadrant.id,
            "quadrant": "DROP" if quadrant is None else quadrant.name,
            "original_goemotions": original_labels,
        }

    mapped = dataset.map(add_quadrant)
    mapped = mapped.filter(lambda example: example["emotion_id"] >= 0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)

    tokenized = mapped.map(tokenize, batched=True)
    for split in tokenized:
        keep_columns = {"input_ids", "attention_mask", "token_type_ids", "emotion_id"}
        remove_columns = [column for column in tokenized[split].column_names if column not in keep_columns]
        tokenized[split] = tokenized[split].remove_columns(remove_columns)
        tokenized[split] = tokenized[split].rename_column("emotion_id", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOPIA_QUADRANTS),
        id2label={quadrant.id: quadrant.name for quadrant in EMOPIA_QUADRANTS},
        label2id={quadrant.name: quadrant.id for quadrant in EMOPIA_QUADRANTS},
    )

    training_config = config.get("training", {})
    output_dir = resolve_path(training_config.get("output_dir", "artifacts/classifier"))
    output_dir.mkdir(parents=True, exist_ok=True)
    args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training_config.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 16)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 32)),
        "learning_rate": float(training_config.get("learning_rate", 2e-5)),
        "weight_decay": float(training_config.get("weight_decay", 0.01)),
        "seed": int(training_config.get("seed", 1508)),
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
    }
    try:
        training_args = TrainingArguments(**args_kwargs)
    except TypeError:
        args_kwargs["eval_strategy"] = args_kwargs.pop("evaluation_strategy")
        training_args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_classifier_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(tokenized["test"]) if "test" in tokenized else trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({quadrant.name: quadrant.id for quadrant in EMOPIA_QUADRANTS}, indent=2),
        encoding="utf-8",
    )


def compute_classifier_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
