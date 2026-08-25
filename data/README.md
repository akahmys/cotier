# 📊 Cotier Datasets & Preprocessing Strategy

This directory contains preprocessed and cached datasets for Cotier's 3-phase training pipeline.

---

## 📚 1. Target Datasets & License Audit

| Phase | Purpose | Dataset | Source / License | Size / Subset |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | 日英・コード基底埋め込み | `roneneldan/TinyStories` | CDLA-Permissive / Apache-2.0 | ~2M stories |
| | | `TinyStories-ja` | CC-BY 4.0 / Apache-2.0 互換 | ~1M stories |
| | | `bigcode/the-stack-smol` | Permissive subsets (MIT/Apache/BSD) | ~500k code & JSON samples |
| **Phase 1** | 構造・潜在再帰推論 ($k=1\dots6$) | `fchollet/arc-agi` | Apache-2.0 | 1,000 tasks |
| | | `kyubyong/sudoku` | MIT | ~3M grids |
| | | `Synthetic 2D Maze` | Generated in-repo (MIT) | 500k mazes |
| **Phase 2** | 日英SFT ＆ MCP/Tool-Use | `databricks/databricks-dolly-15k` | CC-BY-SA 3.0 / Apache-2.0 | 15k conversations |
| | | `kunishou/databricks-dolly-15k-ja`| CC-BY-SA 3.0 | 15k conversations |
| | | `glaiveai/glaive-function-calling-v2` | Apache-2.0 | 110k MCP / Tool calls |
| | | `gsm8k` & `japanese-gsm8k` | MIT | ~8k reasoning steps |

---

## 🛠️ 2. Automated Download & Preprocessing

```bash
# Download datasets from Hugging Face
python train/scripts/01_download_data.py

# Tokenize and format into ChatML / Tool-Call JSON
python train/scripts/02_preprocess.py
```
