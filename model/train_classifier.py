# Training script for fine-tuning DistilBERT on the HC3 dataset.

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

from config import AppConfig


load_dotenv()

config = AppConfig()
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed used by all supported random number generators.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed()


def _first_non_empty_answer(answers: list[str] | None) -> str | None:
    """Return the first non-empty answer from an HC3 answer list.

    Args:
        answers: List of candidate answers from one HC3 record.

    Returns:
        The first non-empty stripped answer, or None when no valid text exists.
    """

    if not answers:
        return None

    for answer in answers:
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    return None


def load_and_prepare_dataset() -> tuple[list[str], list[int]]:
    """Load HC3, extract balanced human and AI samples, and shuffle them.

    Returns:
        A tuple containing shuffled texts and integer labels, where 0 is human
        text and 1 is AI-generated text.
    """

    raw_dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    records = raw_dataset["train"] if isinstance(raw_dataset, DatasetDict) else raw_dataset

    human_texts: list[str] = []
    ai_texts: list[str] = []

    for item in records:
        human_answer = _first_non_empty_answer(item.get("human_answers"))
        ai_answer = _first_non_empty_answer(item.get("chatgpt_answers"))

        if human_answer and len(human_answer) >= 50:
            human_texts.append(human_answer)

        if ai_answer and len(ai_answer) >= 50:
            ai_texts.append(ai_answer)

    sample_count = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:sample_count]
    ai_texts = ai_texts[:sample_count]

    texts = human_texts + ai_texts
    labels = [0] * sample_count + [1] * sample_count

    paired_samples = list(zip(texts, labels, strict=True))
    random.Random(SEED).shuffle(paired_samples)

    shuffled_texts = [text for text, _ in paired_samples]
    shuffled_labels = [label for _, label in paired_samples]

    print(f"Total samples: {len(shuffled_texts)}")
    print(f"Class distribution: human={shuffled_labels.count(0)}, ai={shuffled_labels.count(1)}")

    return shuffled_texts, shuffled_labels


def tokenize_dataset(texts: list[str], labels: list[int]) -> DatasetDict:
    """Tokenize text samples and split them into train and validation datasets.

    Args:
        texts: Input text samples to tokenize.
        labels: Integer labels aligned with the text samples.

    Returns:
        A HuggingFace DatasetDict with "train" and "val" splits.
    """

    tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    ).shuffle(seed=SEED)

    train_size = int(len(dataset) * config.train_ratio)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})
    dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset_dict


def train_model(dataset: DatasetDict) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer, dict[str, list[float]]]:
    """Fine-tune DistilBERT and save the best validation checkpoint.

    Args:
        dataset: Tokenized HuggingFace DatasetDict with "train" and "val" splits.

    Returns:
        A tuple containing the best model, tokenizer, and training history.
    """

    saved_model_path = Path(config.saved_model_path)
    saved_model_path.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    model.to(config.device)

    train_loader = DataLoader(dataset["train"], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    total_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_training_steps,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in progress_bar:
            batch = {key: value.to(config.device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(train_loss=loss.item())

        average_train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(config.device) for key, value in batch.items()}
                labels = batch["labels"]
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=1)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        val_accuracy = correct_predictions / max(total_predictions, 1)
        history["train_loss"].append(average_train_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={average_train_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(saved_model_path)
            tokenizer.save_pretrained(saved_model_path)
            print(f"Saved best model to {saved_model_path}")

    best_model = DistilBertForSequenceClassification.from_pretrained(saved_model_path)
    best_model.to(config.device)

    return best_model, tokenizer, history


def plot_training_history(history: dict[str, list[float]]) -> None:
    """Plot training loss and validation accuracy by epoch.

    Args:
        history: Dictionary containing "train_loss" and "val_accuracy" lists.

    Returns:
        None.
    """

    assets_path = Path("assets")
    assets_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["val_accuracy"], marker="o")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    figure.tight_layout()
    figure.savefig(assets_path / "training_history.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    texts, labels = load_and_prepare_dataset()
    tokenized_dataset = tokenize_dataset(texts, labels)
    trained_model, trained_tokenizer, training_history = train_model(tokenized_dataset)
    plot_training_history(training_history)

    final_val_accuracy = training_history["val_accuracy"][-1] if training_history["val_accuracy"] else 0.0
    print(f"Saved model path: {config.saved_model_path}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
