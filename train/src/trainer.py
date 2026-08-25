"""Unified CotierTrainer for phase-based training and fine-tuning."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TokenizedDataset
from src.export import export_model_artifacts
from src.loss import CotierJointLoss
from src.model import CotierConfig, CotierForCausalLM

logger = logging.getLogger("cotier_trainer")


@dataclass
class TrainingArgs:
    """Training hyperparameters and configuration."""

    data_path: str | Path
    output_dir: str | Path
    epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    max_steps: int = -1
    max_seq_len: int = 256
    lambda_pred: float = 0.1
    lambda_ponder: float = 0.01
    beta_ponder: float = 0.2
    max_grad_norm: float = 1.0
    max_cycles: int | None = None
    num_workers: int = 0


class CotierTrainer:
    """Unified trainer for Cotier cortical column architectures.

    Encapsulates optimizer creation, cosine learning rate scheduling,
    loss computation with PonderNet regularization, gradient clipping,
    and automatic checkpoint exporting.
    """

    def __init__(
        self,
        config: CotierConfig,
        model: CotierForCausalLM,
        args: TrainingArgs,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.args = args

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.loss_fn = CotierJointLoss(
            lambda_pred_error=self.args.lambda_pred,
            lambda_ponder=self.args.lambda_ponder,
            lambda_geom=self.args.beta_ponder,
        )

    def _create_optimizer_and_scheduler(
        self, total_steps: int
    ) -> tuple[AdamW, LambdaLR]:
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.args.lr)

        def lr_lambda(current_step: int) -> float:
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            progress = float(current_step - self.args.warmup_steps) / float(
                max(1, total_steps - self.args.warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def train(self) -> None:
        """Executes the full training loop."""
        logger.info("Initializing training on device: %s", self.device)
        dataset = TokenizedDataset(
            self.args.data_path,
            max_seq_len=self.args.max_seq_len,
            pad_id=self.config.pad_token_id,
        )
        if len(dataset) == 0:
            logger.warning("Dataset is empty. Exiting.")
            return

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )

        total_steps = len(loader) * self.args.epochs
        if self.args.max_steps > 0:
            total_steps = min(total_steps, self.args.max_steps)

        optimizer, scheduler = self._create_optimizer_and_scheduler(total_steps)

        self.model.train()
        global_step = 0
        running_loss = 0.0

        pbar = tqdm(total=total_steps, desc="Training")
        for epoch in range(self.args.epochs):
            for batch in loader:
                # Shape: (B, L)
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, max_cycles=self.args.max_cycles)
                losses = self.loss_fn(outputs, targets)
                total_loss = losses.total_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()

                running_loss += total_loss.item()
                global_step += 1

                pbar.set_postfix(
                    epoch=epoch + 1,
                    loss=f"{total_loss.item():.4f}",
                    task=f"{losses.task_loss.item():.4f}",
                    ponder=f"{losses.ponder_loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                pbar.update(1)

                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    break
            if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                break

        pbar.close()
        avg_loss = running_loss / max(1, global_step)
        logger.info("Training completed. Final average loss: %.4f", avg_loss)

        # Export artifacts
        logger.info("Exporting model artifacts to %s...", self.args.output_dir)
        export_model_artifacts(
            self.model,
            self.args.output_dir,
        )
        logger.info("Training & export completed successfully.")
