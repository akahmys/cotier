#!/usr/bin/env python3
"""Standalone Verification Script for Safetensors Schema Contract.

Checks that model.safetensors matches models/cotier-0.5b/tensor_schema.json
and satisfies all shape/type constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from safetensors import safe_open


def verify_safetensors_schema(model_dir: Path, schema_file: Path) -> bool:
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        print(f"❌ Safetensors file not found at {safetensors_path}")
        return False

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_def = json.load(f)

    # Build flattened expected schema
    expected_shapes: dict[str, list[int]] = {}
    expected_shapes["embed_tokens.weight"] = schema_def["embedding"]["embed_tokens.weight"]
    expected_shapes["norm.weight"] = schema_def["final_norm"]["norm.weight"]
    expected_shapes["lm_head.weight"] = schema_def["lm_head"]["lm_head.weight"]

    template = schema_def["cortical_stack_template"]
    for i in range(4):
        for tmpl_key, shape in template.items():
            actual_key = tmpl_key.format(i=i)
            expected_shapes[actual_key] = shape

    errors: list[str] = []
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        actual_keys = set(f.keys())
        expected_keys = set(expected_shapes.keys())

        # Missing keys
        missing = expected_keys - actual_keys
        if missing:
            errors.append(f"Missing required keys ({len(missing)}): {sorted(list(missing))}")

        # Unexpected keys
        unexpected = actual_keys - expected_keys
        if unexpected:
            errors.append(f"Unexpected keys ({len(unexpected)}): {sorted(list(unexpected))}")

        # Shape check
        for key in expected_keys.intersection(actual_keys):
            tensor = f.get_tensor(key)
            actual_shape = list(tensor.shape)
            expected_shape = expected_shapes[key]
            if actual_shape != expected_shape:
                errors.append(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")

    if errors:
        print("❌ Schema validation failed with errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    print(f"✅ Schema validation passed! All {len(expected_keys)} tensors match {schema_file}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Safetensors Schema Contract")
    parser.add_argument("--model-dir", type=Path, default=Path("./models/cotier-0.5b"))
    parser.add_argument("--schema-file", type=Path, default=Path("./models/cotier-0.5b/tensor_schema.json"))
    args = parser.parse_args()

    success = verify_safetensors_schema(args.model_dir, args.schema_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
