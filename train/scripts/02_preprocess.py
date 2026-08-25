"""Data Preprocessing and Tokenizer Generation Script for Cotier-0.45B.

Builds standard 32,000 vocab tokenizer with Japanese, English, Code, and Tool-call special tokens,
and encodes dataset splits into tensor batches for training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("preprocess")

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


def build_and_train_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 32000,
    output_path: str = "./models/cotier-0.5b/tokenizer.json",
) -> Tokenizer:
    """Builds and trains a Byte-Level BPE tokenizer with special tokens."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    logger.info(
        "Training tokenizer with %d special tokens and vocab size %d...",
        len(SPECIAL_TOKENS),
        vocab_size,
    )
    tokenizer.train(files=corpus_files, trainer=trainer)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_file))
    logger.info("Saved tokenizer to %s", out_file)
    return tokenizer


def tokenize_and_cache(
    tokenizer: Tokenizer,
    jsonl_path: str | Path,
    out_path: str | Path,
    max_seq_len: int = 2048,
    text_key: str = "formatted",
) -> int:
    """Tokenizes JSONL file and saves token arrays as JSONL."""
    in_file = Path(jsonl_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(in_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get(text_key) or data.get("text") or data.get("prompt", "")
            encoded = tokenizer.encode(text)
            input_ids = encoded.ids[:max_seq_len]
            records.append({"input_ids": input_ids, "length": len(input_ids)})

    with open(out_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Processed %d items from %s -> %s", len(records), in_file.name, out_file.name)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess data and generate tokenizer for Cotier."
    )
    parser.add_argument("--raw-dir", type=str, default="./data/raw", help="Raw data directory")
    parser.add_argument(
        "--processed-dir", type=str, default="./data/processed", help="Processed data directory"
    )
    parser.add_argument(
        "--tokenizer-out",
        type=str,
        default="./models/cotier-0.5b/tokenizer.json",
        help="Tokenizer output path",
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(raw_dir.glob("*.jsonl"))
    if not raw_files:
        logger.error(
            "No raw jsonl files found in %s. Please run 01_download_data.py first.", raw_dir
        )
        return

    # Train tokenizer on raw files
    str_raw_files = [str(f) for f in raw_files]
    tokenizer = build_and_train_tokenizer(
        corpus_files=str_raw_files,
        vocab_size=args.vocab_size,
        output_path=args.tokenizer_out,
    )

    # Process each phase file
    for raw_file in raw_files:
        out_name = f"tokenized_{raw_file.stem}.jsonl"
        tokenize_and_cache(tokenizer, raw_file, processed_dir / out_name)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
