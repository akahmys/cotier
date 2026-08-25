"""Unified Tokenized Dataset implementation for Cotier training phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class TokenizedBatch(TypedDict):
    """Batch dictionary returned by TokenizedDataset."""

    input_ids: torch.Tensor
    targets: torch.Tensor


class TokenizedDataset(Dataset[TokenizedBatch]):
    """Dataset loading preprocessed token sequences from JSONL.

    Shapes:
        - input_ids: (L,) int64 tensor padded to max_seq_len
        - targets:   (L,) int64 tensor for next-token prediction
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        max_seq_len: int = 256,
        pad_id: int = 2,
    ) -> None:
        self.samples: list[list[int]] = []
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                data = json.loads(line_str)
                ids: list[int] = data.get("input_ids", [])
                if len(ids) > 1:
                    self.samples.append(ids[:max_seq_len])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TokenizedBatch:
        ids = self.samples[idx]
        seq_len = len(ids)

        if seq_len < self.max_seq_len:
            padded = ids + [self.pad_id] * (self.max_seq_len - seq_len)
        else:
            padded = ids[: self.max_seq_len]

        tensor_ids = torch.tensor(padded, dtype=torch.long)
        return {"input_ids": tensor_ids, "targets": tensor_ids.clone()}
