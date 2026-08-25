"""Phase 1: Structural Reasoning and Adaptive Latent Recurrence Training.

Trains Layer V recurrent loops (k=1..6) and Layer VI PonderNet Halting Unit
using structured grid reasoning datasets (Sudoku, Maze, ARC-AGI).
"""

from __future__ import annotations

import argparse
import logging

from src.model import CotierConfig, CotierForCausalLM
from src.trainer import CotierTrainer, TrainingArgs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_phase1")


def train_phase1(
    data_path: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    epochs: int = 1,
    max_steps: int = -1,
) -> None:
    """Executes Phase 1 structural reasoning and PonderNet training loop via CotierTrainer."""
    logger.info("Starting Phase 1 structural reasoning training...")
    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=6,
        ponder_lambda_geom=0.2,
    )
    model = CotierForCausalLM(config)

    args = TrainingArgs(
        data_path=data_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        max_steps=max_steps,
        max_seq_len=128,
        lambda_pred=0.1,
        lambda_ponder=0.01,
        beta_ponder=0.2,
        max_cycles=6,
    )

    trainer = CotierTrainer(config=config, model=model, args=args)
    trainer.train()
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

