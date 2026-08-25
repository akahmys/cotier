"""Phase 2.5: Safety, Ethics & Quality Alignment Training.

Fine-tunes the model to deliver polite, constructive refusals for hazardous
requests while maintaining general conversational and tool capabilities.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from safetensors.torch import load_file

from src.model import CotierConfig, CotierForCausalLM
from src.trainer import CotierTrainer, TrainingArgs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_safety")


def train_safety(
    data_path: str,
    output_dir: str,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    epochs: int = 2,
    max_steps: int = -1,
) -> None:
    """Executes Phase 2.5 safety & ethical alignment training loop."""
    logger.info("Starting Phase 2.5 Safety & Alignment Training...")
    out_dir = Path(output_dir)
    safetensors_path = out_dir / "model.safetensors"

    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=6,
    )
    model = CotierForCausalLM(config)

    # Load existing checkpoint if present
    if safetensors_path.exists():
        logger.info("Loading existing checkpoint from %s...", safetensors_path)
        state_dict = load_file(str(safetensors_path))
        # Handle tied embeddings
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded base weights successfully.")

    args = TrainingArgs(
        data_path=data_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        max_steps=max_steps,
        max_seq_len=128,
        lambda_pred=0.08,
        lambda_ponder=0.01,
        beta_ponder=0.2,
        max_cycles=6,
    )

    trainer = CotierTrainer(config=config, model=model, args=args)
    trainer.train()
    logger.info("Phase 2.5 Safety Alignment Training completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5 Safety alignment for Cotier-0.45B")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/processed/tokenized_phase2_5_alignment.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="./models/cotier-0.5b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    train_safety(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
