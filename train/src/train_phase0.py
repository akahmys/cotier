"""Phase 0: Bilingual and Code Base Representation Pre-training.

Bootstraps 32,000 token vocabulary embeddings and foundational syntax
using TinyStories (EN/JA) and The Stack Smol with single-cycle recurrence.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.export import export_model_artifacts
from src.loss import CotierJointLoss
from src.model import CotierConfig, CotierForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_phase0")


class TokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset loading preprocessed token sequences."""

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 256, pad_id: int = 2) -> None:
        self.samples: list[list[int]] = []
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                ids = data["input_ids"]
                if len(ids) > 1:
                    self.samples.append(ids[:max_seq_len])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx]
        seq_len = len(ids)
        if seq_len < self.max_seq_len:
            padded = ids + [self.pad_id] * (self.max_seq_len - seq_len)
        else:
            padded = ids[: self.max_seq_len]

        tensor_ids = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": tensor_ids, "targets": tensor_ids.clone()}


def train_phase0(
    data_path: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    epochs: int = 1,
    max_steps: int | None = 50,
) -> None:
    """Executes Phase 0 base embedding training loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # Initialize model config and model
    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=1,  # Fast pre-training cycle for Phase 0
    )
    model = CotierForCausalLM(config).to(device)
    loss_fn = CotierJointLoss(lambda_pred_error=0.05, lambda_ponder=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    dataset = TokenizedDataset(data_path, max_seq_len=128)
    if len(dataset) == 0:
        logger.warning("Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    step = 0
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, max_cycles=1)
            losses = loss_fn(output, targets)

            losses.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            progress.set_postfix(
                {
                    "loss": f"{losses.total_loss.item():.4f}",
                    "task_ce": f"{losses.task_loss.item():.4f}",
                }
            )

            if max_steps is not None and max_steps > 0 and step >= max_steps:
                logger.info("Reached maximum step limit (%d). Finishing Phase 0.", max_steps)
                break
        if max_steps is not None and max_steps > 0 and step >= max_steps:
            break

    # Export checkpoint
    logger.info("Exporting Phase 0 checkpoint to %s...", output_dir)
    export_model_artifacts(model, output_dir)
    logger.info("Phase 0 training successfully completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 training for Cotier-0.45B")
    parser.add_argument(
        "--data-path", type=str, default="./data/processed/tokenized_phase0_embedding.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default="./models/cotier-0.5b")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    train_phase0(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
