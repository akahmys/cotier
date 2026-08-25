#!/usr/bin/env python3
"""Evaluation probe to test generation capabilities of Cotier-0.45B."""

import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Add train directory to path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "train"))

from src.model import CotierConfig, CotierForCausalLM

def main():
    model_dir = project_root / "models" / "cotier-0.5b"
    tokenizer_path = model_dir / "tokenizer.json"
    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    config = CotierConfig(
        vocab_size=cfg.get("vocab_size", 32000),
        hidden_size=cfg.get("hidden_size", 1024),
        intermediate_size=cfg.get("intermediate_size", 2816),
        num_attention_heads=cfg.get("num_attention_heads", 16),
        num_cortical_stacks=cfg.get("num_cortical_stacks", 4),
        max_recurrent_cycles=cfg.get("max_recurrent_cycles", 6),
    )

    model = CotierForCausalLM(config).eval()
    state_dict = load_file(str(safetensors_path))
    model.load_state_dict(state_dict, strict=False)

    prompts = [
        "<|im_start|>user\nこんにちは、あなたの名前は？<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a python function to add two numbers.<|im_end|>\n<|im_start|>assistant\n",
        '<|im_start|>system\nTools: [{"type": "function", "function": {"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"}}}}}]<|im_end|>\n<|im_start|>user\n東京の天気を教えて<|im_end|>\n<|im_start|>assistant\n',
    ]

    print("=" * 60)
    print("🧠 Cotier-0.45B Generation Probe")
    print("=" * 60)

    for p in prompts:
        encoded = tokenizer.encode(p)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)
        
        # Prefill & 10 decode steps
        generated = list(encoded.ids)
        print(f"\n📝 Prompt: {repr(p.strip())}")
        
        with torch.no_grad():
            curr_input = input_ids
            for step in range(12):
                output = model(curr_input)
                next_token_logits = output.logits[0, -1, :]
                next_token = int(torch.argmax(next_token_logits).item())
                generated.append(next_token)
                curr_input = torch.tensor([generated], dtype=torch.long)
                if next_token == tokenizer.token_to_id("<|im_end|>"):
                    break
        
        gen_text = tokenizer.decode(generated[len(encoded.ids):])
        print(f"🤖 Output: {gen_text}")
        print(f"⚡ PonderNet Halting Probs (last token): {[round(x, 4) for x in output.halting_probs[0, -1, :].tolist()]}")

if __name__ == "__main__":
    main()
