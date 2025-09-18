"""Minimal GPT-style language-model training loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)


class TokenDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """A tiny JSONL-backed dataset returning input/target token tensors."""

    def __init__(self, path: Path, seq_len: int) -> None:
        with path.open("r", encoding="utf-8") as fh:
            self.items = [json.loads(line)["tokens"] for line in fh]
        if not self.items:
            raise RuntimeError(f"Dataset at {path} is empty.")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.items[index][: self.seq_len + 1]
        if len(tokens) < 2:
            tokens = tokens + [0] * (2 - len(tokens))
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        return inputs, targets


def collate_batch(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


def train(config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    vocab_path = Path(config["model"]["vocab_path"])
    with vocab_path.open("r", encoding="utf-8") as fh:
        vocab = json.load(fh)
    vocab_size = max(vocab.values()) + 1

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            n_layer=config["model"]["n_layer"],
            n_head=config["model"]["n_head"],
            n_embd=config["model"]["n_embd"],
            n_positions=config["train"]["seq_len"],
        )
    )

    dataset = TokenDataset(Path(config["data"]["train_jsonl"]), config["train"]["seq_len"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["train"]["lr"])
    total_steps = len(dataloader) * config["train"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    torch.manual_seed(42)
    model.train()

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(config["train"]["epochs"]):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"epoch {epoch + 1} loss={avg_loss:.4f}")
        model.save_pretrained(checkpoints_dir / f"epoch{epoch + 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/training/configs/default.yaml"),
        help="Path to the training configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
