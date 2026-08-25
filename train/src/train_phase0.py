"""Phase 0: Bilingual and Code Base Representation Pre-training.

Bootstraps 32,000 token vocabulary embeddings and foundational syntax
using TinyStories (EN/JA) and The Stack Smol with single-cycle recurrence.
"""

from __future__ import annotations

import argparse
import logging

from src.model import CotierConfig, CotierForCausalLM
from src.trainer import CotierTrainer, TrainingArgs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_phase0")


def train_phase0(
    data_path: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    epochs: int = 1,
    max_steps: int = -1,
) -> None:
    """Executes Phase 0 base embedding training loop via CotierTrainer."""
    logger.info("Starting Phase 0 base embedding training...")
    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=1,
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
        lambda_pred=0.05,
        lambda_ponder=0.0,
        max_cycles=1,
    )

    trainer = CotierTrainer(config=config, model=model, args=args)
    trainer.train()
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

