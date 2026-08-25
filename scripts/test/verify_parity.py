#!/usr/bin/env python3
"""Cross-Framework Numerical Parity Verification Script.

Verifies that PyTorch CotierForCausalLM and Rust candle CorticalModel
produce numerically identical output logits (max error < 1e-4)
given the exact same model weights and input token sequence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add train directory to pythonpath
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "train"))

from src.model import CotierConfig, CotierForCausalLM  # noqa: E402


def verify_parity(model_dir: Path, tokens: list[int], tolerance: float = 1e-4) -> bool:
    print("=" * 60)
    print("🔬 Running PyTorch vs Rust Candle Numerical Parity Check")
    print(f"📂 Model Directory: {model_dir}")
    print(f"🔤 Input Tokens: {tokens}")
    print(f"🎯 Target Tolerance: L_inf < {tolerance}")
    print("=" * 60)

    # 1. Load PyTorch Model
    config_path = model_dir / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = CotierConfig(
        vocab_size=config_dict.get("vocab_size", 32000),
        hidden_size=config_dict.get("hidden_size", 1024),
        intermediate_size=config_dict.get("intermediate_size", 2816),
        num_attention_heads=config_dict.get("num_attention_heads", 16),
        num_key_value_heads=config_dict.get("num_key_value_heads", 16),
        head_dim=config_dict.get("head_dim", 64),
        num_cortical_stacks=config_dict.get("num_cortical_stacks", 4),
        max_recurrent_cycles=config_dict.get("max_recurrent_cycles", 6),
        recurrent_alpha=config_dict.get("recurrent_alpha", 0.1),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        rope_theta=config_dict.get("rope_theta", 10000.0),
    )

    pt_model = CotierForCausalLM(config).eval().to(torch.float32)
    safetensors_path = model_dir / "model.safetensors"
    state_dict = load_file(str(safetensors_path))
    pt_model.load_state_dict(state_dict, strict=False)

    # 2. PyTorch Forward Pass (Prefill k=1)
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        output = pt_model(input_tensor, is_prefill=True)
        pt_logits = output.logits[0].cpu().numpy()  # [L, V]

    # 3. Rust Forward Pass
    tokens_str = ",".join(str(t) for t in tokens)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as tmp_out:
        tmp_path = Path(tmp_out.name)

    try:
        cmd = [
            "cargo",
            "run",
            "--manifest-path",
            str(project_root / "server" / "Cargo.toml"),
            "--release",
            "--",
            "eval-logits",
            "--model",
            str(model_dir),
            "--tokens",
            tokens_str,
            "--out",
            str(tmp_path),
        ]
        print(f"▶️ Executing Rust inference: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        with open(tmp_path, "r", encoding="utf-8") as f:
            rust_logits_raw = json.load(f)
            rust_logits = rust_logits_raw[0]  # [L, V]

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # 4. Compare Logits
    pt_flat = pt_logits.flatten()
    rust_tensor = torch.tensor(rust_logits, dtype=torch.float32)
    rust_flat = rust_tensor.numpy().flatten()

    abs_diff = abs(pt_flat - rust_flat)
    max_error = float(abs_diff.max())
    mean_error = float(abs_diff.mean())

    print("-" * 60)
    print(f"📊 Maximum Absolute Error (L_inf): {max_error:.6e}")
    print(f"📊 Mean Absolute Error (L_1):      {mean_error:.6e}")
    print("-" * 60)

    if max_error < tolerance:
        print(f"✅ Numerical Parity PASSED! (L_inf = {max_error:.6e} < {tolerance})")
        return True
    else:
        print(f"❌ Numerical Parity FAILED! Max error {max_error:.6e} exceeds tolerance {tolerance}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch vs Rust Parity Check")
    parser.add_argument("--model-dir", type=Path, default=Path("./models/cotier-0.5b"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--tokens", type=str, default="1,42,105,2048,150,2")
    args = parser.parse_args()

    token_list = [int(t.strip()) for t in args.tokens.split(",") if t.strip()]
    success = verify_parity(args.model_dir, token_list, args.tolerance)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
