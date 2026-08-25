"""Phase 1: Structural Reasoning and Adaptive Latent Recurrence Training.

Trains Layer V recurrent loops (k=1..6) and Layer VI PonderNet Halting Unit
using structured grid reasoning datasets (Sudoku, Maze, ARC-AGI).
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.export import export_model_artifacts
from src.loss import CotierJointLoss
from src.model import CotierConfig, CotierForCausalLM
from src.train_phase0 import TokenizedDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_phase1")


def train_phase1(
    data_path: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 1,
    max_steps: int | None = 50,
) -> None:
    """Executes Phase 1 structural reasoning and PonderNet training loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=6,
        ponder_lambda_geom=0.2,
    )
    model = CotierForCausalLM(config).to(device)
    loss_fn = CotierJointLoss(lambda_pred_error=0.1, lambda_ponder=0.01, lambda_geom=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    dataset = TokenizedDataset(data_path, max_seq_len=128)
    if len(dataset) == 0:
        logger.warning("Dataset is empty. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    step = 0
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Phase 1 Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, max_cycles=6)
            losses = loss_fn(output, targets)

            losses.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            progress.set_postfix(
                {
                    "loss": f"{losses.total_loss.item():.4f}",
                    "task": f"{losses.task_loss.item():.4f}",
                    "ponder": f"{losses.ponder_loss.item():.4f}",
                }
            )

            if max_steps is not None and max_steps > 0 and step >= max_steps:
                logger.info("Reached maximum step limit (%d). Finishing Phase 1.", max_steps)
                break
        if max_steps is not None and max_steps > 0 and step >= max_steps:
            break

    logger.info("Exporting Phase 1 checkpoint to %s...", output_dir)
    export_model_artifacts(model, output_dir)
    logger.info("Phase 1 training successfully completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 training for Cotier-0.45B")
    parser.add_argument(
        "--data-path", type=str, default="./data/processed/tokenized_phase1_structural.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default="./models/cotier-0.5b")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    train_phase1(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
