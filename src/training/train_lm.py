import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.optimization import get_cosine_schedule_with_warmup


class DS(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str, seq: int = 2048) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            self.items = [json.loads(line)["tokens"] for line in handle]
        self.seq = seq

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.items[index][: self.seq]
        if len(tokens) <= 2:
            tokens = tokens + [0, 0]
        input_tokens = torch.tensor(tokens[:-1])
        target_tokens = torch.tensor(tokens[1:])
        return input_tokens, target_tokens


def collate(
    batch: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)
    return (
        torch.nn.utils.rnn.pad_sequence(list(inputs), True, 0),
        torch.nn.utils.rnn.pad_sequence(list(targets), True, 0),
    )


def main(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    with open(cfg["model"]["vocab_path"], "r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    vocab_size = max(vocab.values()) + 1
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            n_layer=cfg["model"]["n_layer"],
            n_head=cfg["model"]["n_head"],
            n_embd=cfg["model"]["n_embd"],
            n_positions=cfg["train"]["seq_len"],
        )
    )
    dataset = DS(cfg["data"]["train_jsonl"], cfg["train"]["seq_len"])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, len(dataloader) * cfg["train"]["epochs"]
    )
    Path("checkpoints").mkdir(exist_ok=True)
    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        loss_sum = 0.0
        steps = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(input_ids=inputs, labels=targets)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            steps += 1
        mean_loss = loss_sum / max(steps, 1)
        print(f"epoch {epoch + 1} loss={mean_loss:.4f}")
        model.save_pretrained(f"checkpoints/epoch{epoch + 1}")
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/training/configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
