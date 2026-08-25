"""Master Automated Initial Education Orchestration Pipeline for Cotier-0.45B.

Executes end-to-end dataset generation/downloading, tokenization,
Phase 0 -> Phase 1 -> Phase 2 sequential training on Apple Silicon MPS,
model artifacts export, tensor schema audit, and parity validation.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("education_pipeline")

SCALE_CONFIGS = {
    "smoke": {
        "sample_size": 200,
        "p0_epochs": 1,
        "p1_epochs": 1,
        "p2_epochs": 1,
        "batch_size": 4,
    },
    "medium": {
        "sample_size": 2000,
        "p0_epochs": 3,
        "p1_epochs": 3,
        "p2_epochs": 3,
        "batch_size": 8,
    },
    "full": {
        "sample_size": 10000,
        "p0_epochs": 5,
        "p1_epochs": 5,
        "p2_epochs": 5,
        "batch_size": 8,
    },
}


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    cmd_str = " ".join(cmd)
    logger.info("▶️ Executing: %s", cmd_str)
    start = time.time()
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    duration = time.time() - start

    if res.returncode != 0:
        logger.error(
            "❌ Command failed (code %d) in %.2fs:\n%s\n%s",
            res.returncode,
            duration,
            res.stdout,
            res.stderr,
        )
        raise RuntimeError(f"Command failed: {cmd_str}")

    logger.info("✅ Finished in %.2fs", duration)
    if res.stdout:
        last_lines = "\n".join(res.stdout.strip().split("\n")[-5:])
        logger.info("Tail output:\n%s", last_lines)


def run_pipeline(scale: str, output_dir: Path, data_dir: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    train_dir = project_root / "train"
    cfg = SCALE_CONFIGS[scale]

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    print("=" * 70)
    print(f"🚀 Starting Cotier Automated Education Pipeline (Scale: {scale.upper()})")
    print(f"📂 Output Model Directory: {output_dir}")
    print(f"📊 Config: {cfg}")
    print("=" * 70)

    # 1. Download / Synthesize Data
    logger.info("[Step 1/6] Ingesting and generating datasets...")
    run_command(
        [
            sys.executable,
            "scripts/01_download_data.py",
            "--output-dir",
            str(raw_dir),
            "--sample-size",
            str(cfg["sample_size"]),
        ],
        cwd=train_dir,
    )

    # 2. Tokenize & Preprocess Data
    logger.info("[Step 2/6] Building BPE tokenizer and preprocessing splits...")
    tokenizer_out = output_dir / "tokenizer.json"
    run_command(
        [
            sys.executable,
            "scripts/02_preprocess.py",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--tokenizer-out",
            str(tokenizer_out),
        ],
        cwd=train_dir,
    )

    # 3. Phase 0: Base Embedding Bootstrap
    logger.info("[Step 3/6] Running Phase 0: Bilingual & Code Base Representation Pre-training...")
    p0_data = processed_dir / "tokenized_phase0_embedding.jsonl"
    run_command(
        [
            sys.executable,
            "-m",
            "src.train_phase0",
            "--data-path",
            str(p0_data),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(cfg["p0_epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            "3e-4",
            "--max-steps",
            "-1",
        ],
        cwd=train_dir,
    )

    # 4. Phase 1: Structural Reasoning
    logger.info("[Step 4/6] Running Phase 1: Structural Reasoning & PonderNet Latent Recurrence...")
    p1_data = processed_dir / "tokenized_phase1_structural.jsonl"
    run_command(
        [
            sys.executable,
            "-m",
            "src.train_phase1",
            "--data-path",
            str(p1_data),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(cfg["p1_epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            "1e-4",
            "--max-steps",
            "-1",
        ],
        cwd=train_dir,
    )

    # 5. Phase 2: Bilingual SFT & MCP Tool-Use
    logger.info("[Step 5/6] Running Phase 2: Bilingual SFT & MCP Tool-Use Alignment...")
    p2_data = processed_dir / "tokenized_phase2_sft_mcp.jsonl"
    run_command(
        [
            sys.executable,
            "-m",
            "src.train_phase2",
            "--data-path",
            str(p2_data),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(cfg["p2_epochs"]),
            "--batch-size",
            str(cfg["batch_size"]),
            "--lr",
            "5e-5",
            "--max-steps",
            "-1",
        ],
        cwd=train_dir,
    )

    # 6. Post-Training Validation Audits
    logger.info("[Step 6/6] Running Safetensors Schema Contract and Numerical Parity Checks...")
    schema_file = output_dir / "tensor_schema.json"
    run_command(
        [
            sys.executable,
            "../scripts/test/verify_tensor_schema.py",
            "--model-dir",
            str(output_dir),
            "--schema-file",
            str(schema_file),
        ],
        cwd=train_dir,
    )

    run_command(
        [
            sys.executable,
            "../scripts/test/verify_parity.py",
            "--model-dir",
            str(output_dir),
            "--tolerance",
            "1e-4",
        ],
        cwd=train_dir,
    )

    print("=" * 70)
    print("🎉 Cotier Automated Education Pipeline COMPLETED SUCCESSFULLY!")
    print(f"📂 Model Checkpoint Ready: {output_dir}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Automated Education Pipeline for Cotier-0.45B"
    )
    parser.add_argument(
        "--scale",
        type=str,
        choices=["smoke", "medium", "full"],
        default="medium",
        help="Training scale preset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./models/cotier-0.5b"),
        help="Output model directory",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Data directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    d_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir

    run_pipeline(args.scale, out_dir, d_dir)


if __name__ == "__main__":
    main()
