# Cotier (Cortical-Tier) 開発ロードマップ (ROADMAP.md)

本ロードマップは、Apple Siliconネイティブの皮質階層推論エンジン **`Cotier` (0.45B / 4スタック皮質コラム)** の開発全体計画、マイルストーン、各フェーズの具体的タスクおよび成功基準を定義する。

---

## 🗺️ 全体ロードマップ概要

```mermaid
gantt
    title Cotier 開発ロードマップ
    dateFormat  YYYY-MM-DD
    section v1.0 MVP (日英/コード/MCP・対話・自己成長)
    Sprint 1: 環境セットアップ & 基盤構築         :done, s1, 2026-08-25, 3d
    Sprint 2: Python 初期教育 (日英・コード・MCP)  :done, s2, after s1, 6d
    Sprint 3: Rust 推論コア & Metal 最適化       :done, s3, after s2, 4d
    Sprint 4: OpenAI / MCP サーバー & 記憶固定化 :done, s4, after s3, 5d
    Sprint 5: E2E 結合検証 & エージェント接続   :done, s5, after s4, 3d
    Sprint 6: コードベース総合リファクタリング   :done, s6, after s5, 2d
    Sprint 7: 倫理・安全性アライメント & 免疫ガードレール :done, s7, after s6, 3d
    
    section v1.1 高速化 (小脳ショートカット)
    小脳 Fast-Path & Metal カーネル融合           :active, v11, after s7, 5d
    
    section v1.2 自律制御 (神経修飾)
    Surprise 駆動の動的 Temperature / 覚醒制御   :v12, after v11, 4d
    
    section v2.0 高度推論 (視床 & 大脳基底核)
    視床トークン間引き & 基底核 Go/No-Go 探索    :v20, after v12, 10d
```

---

## 🚀 Phase 1: Cotier v1.0 (MVP 開発計画)

**目標**: 日英・コード・MCPツール呼び出し能力を備えたベースモデル (0.45B) を生成し、Rust 推論サーバーを起動して、Cursor / Continue / Open-WebUI と会話・ツール実行しながらモデルが自己成長する垂直統合システムを完成させる。

---

### Sprint 1: プロジェクト基盤 & 環境構築 (Days 1〜3)
* [x] **Python 学習環境セットアップ (`train/`)**:
  - `pyproject.toml` 作成 (PyTorch / MLX, Hugging Face `transformers`, `datasets`, `safetensors`, `accelerate`)
  - MPS (Metal Performance Shaders) 加速の動作確認
* [x] **Rust ワークスペース初期化 (`server/`)**:
  - `Cargo.toml` 作成 (`candle-core`, `candle-nn`, `axum`, `tokio`, `rusqlite`, `arc-swap`)
  - Apple Silicon Metal フィーチャーのビルド設定
* [x] **ディレクトリ構造の確定**:
  - `data/`, `train/`, `models/`, `server/` のスケルトン作成

---

### Sprint 2: 初期教育システムの実装 & 学習実行 (`train/`) (Days 4〜9)
* [x] **データ取得 & 前処理スクリプト**:
  - `scripts/01_download_data.py`: 日英TinyStories, The Stack (Code), ARC-AGI, Sudoku 3M, Dolly日英, **Glaive Function Calling v2** の自動DL
  - `scripts/02_preprocess.py`: 日英BPEトークン化、数独グリッド変換、ChatML & Tool-Callフォーマット整形
* [x] **PyTorch 4スタック皮質コラムモデル (`train/src/model.py`)**:
  - 隠れ次元 $D=1024, H=16, D_{\text{ffn}}=2816$、4層スタック構成 (0.45B)
  - Layer I〜VI の順伝播、**RoPE付き因果Self-Attention**、再帰 SwiGLU、**PonderNet 停止分類器（Halting Unit）**の実装
* [x] **3損失複合関数 (`train/src/loss.py`)**:
  - 期待Cross Entropy $\mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{pred\_error}} + \lambda_2 \text{PonderNet KL Loss}$ の実装
* [x] **学習の段階的実行**:
  - **Phase 0 (日英・コード埋め込み)**: TinyStories日英 ＋ The Stackによる 32,000 語彙の基底埋め込み初期化
  - **Phase 1 (構造推論)**: 数独・迷路による Layer V/VI の再帰探索と Early Exit（Ponder Halting）の安定化
  - **Phase 2 (日英SFT ＋ MCP)**: Dolly日英 ＋ Glaive Function Calling による指示追従と正確な JSON Tool Call 生成の学習
* [x] **エクスポート (`train/src/export.py`)**:
  - `model.safetensors`, `config.json`, `tokenizer.json`, `anchors.jsonl` の出力

---

### Sprint 3: Rust 推論コア (`server/src/model.rs`) (Days 10〜13)
* [x] **Safetensors ゼロコピーローダー**:
  - `candle` によるメモリマップド I/O (`mmap`) ロード
* [x] **4スタック皮質フォワードパス & KV Cache (Metal ネイティブ)**:
  - **Prefill 並列処理**: プロンプト処理時の固定ステップ並列実行
  - **Decode Recurrent Step Attention**: 過去トークンの収束 KV Cache ＋ 生成トークンのみの $k=1..6$ ループ
  - In-Place メモリ更新によるゼロアロケーション推論
* [x] **単体テスト & ベンチマーク**:
  - PyTorch 出力との完全一致（数値決定性）テスト
  - tokens/sec およびメモリ消費のベンチマーク測定

---

### Sprint 4: OpenAI / MCP 互換サーバー & 生涯成長ループ (`server/`) (Days 14〜18)
* [x] **axum HTTP / SSE サーバー (`server/src/server.rs`)**:
  - `POST /v1/chat/completions` (OpenAI 互換 SSE ストリーミング ＆ `tools` / `tool_calls` サポート)
  - `GET /v1/models`, `GET /v1/cotier/metrics`, `POST /v1/cotier/feedback`
* [x] **海馬エピソード記憶マネージャー (`server/src/memory.rs`)**:
  - SQLite による高 Surprise 対話ログおよび修正指示の自動永続化
  - 学習対象ガードレール（$+1$ 評価または高 Surprise 対話のみを抽出）の実装
* [x] **睡眠記憶固定化ワーカー (`server/src/learner.rs`)**:
  - アイドル時間検知（5分間リクエストなし）
  - エピソード記憶 ＋ `anchors.jsonl`（30%）の Replay 学習ループ
* [x] **推論優先ロック (Preemption)**:
  - Sleep学習中に推論リクエストが来た場合の即座中断（Yield）・推論優先制御

---

### Sprint 5: E2E 結合検証 & エージェント・MCP 接続 (Days 19〜21)
* [x] **CLI 対話モード (`cotier chat`)**:
  - リアルタイムの思考メーター（Cortical Load: cycles, surprise）描画
* [x] **フロントエンド & MCP ツール実行テスト**:
  - **Cursor / VSCode Continue**: コーディングアシスタント ＆ 外部コマンド実行テスト (`/v1/chat/completions`)
  - **Open-WebUI**: MCP サーバー経由での Web 検索 / 計算ツール呼び出しテスト (`/v1/chat/completions` + `tools`)
* [x] **自己成長テスト**:
  - 会話やツール実行でユーザーが指示したルールを海馬エピソード記憶に永続化し、睡眠学習で定着するパイプラインを検証

---

### Sprint 6: コードベース総合リファクタリング & CR-15 ハードニング (Days 22〜23)
* [x] **Python 学習基盤の DRY 化 & 型強化 (`train/src/`)**:
  - `train/src/dataset.py`: `TokenizedDataset` のモジュール分離・共通化
  - `train/src/trainer.py`: `CotierTrainer` 基盤クラスの構築（Phase 0〜2 の学習・最適化ループ共通化）
  - **CODING 規約適合**: `mypy --strict src/` 完全適合、全テンソル変換への形状注記コメント付与、PonderNet $\epsilon$-clamping ($\epsilon=10^{-7}$) 徹底
* [x] **Rust 推論ジェネレータの共通化 & CR-15 準拠 (`server/src/`)**:
  - `server/src/engine.rs`: `TokenStreamIterator` / `GenerationEngine` の抽出
  - `generate_non_streaming`, `generate_sse_stream`, CLI `chat` の自己回帰ループ（forward・argmax・surprise・decode）一本化
  - **CR-15 規約適合**:
    - **Rule 1 (関数長)**: 標準関数 50行以内、ディスパッチャ 150行以内
    - **Rule 2 & 3 (パニック/Unsafe 禁止)**: `unwrap`/`expect` ゼロ、`unsafe` ゼロ
    - **Rule 4 & 5 (制御フロー/網羅 Match)**: `?` 早期リターン徹底、ドメイン enum ワイルドカード排除
    - **Rule 13 (ゼロアロケーション)**: Decode ループ中の不要なメモリ確保・クローン排除
    - **Rule 14 & 15 (FP32/明示的型指定)**: Surprise / Softmax の FP32 計算、`1.0_f32` リテラル指定
* [x] **KV Cache の型カプセル化 (`server/src/model.rs`)**:
  - `struct LayerKvCache` および `struct ModelKvCache` による安全なキャッシュ管理（**Rule 8 不正状態の排除**）
  - In-Place メモリ更新およびシーケンス長トリミングの型安全化
* [x] **CR-15 / AUDITING / TESTING 全自動回帰検証**:
  - **静的監査 (Rule A1〜A8)**: `bash scripts/audit/verify_compliance.sh` 全項目 PASS
  - **単体・統合テスト (Rule T1)**: `cd train && uv run pytest` & `cd server && cargo test --workspace`
  - **スキーマ & パリティ (Rule T2, T3)**: `verify_tensor_schema.py` & `verify_parity.py` ($L_\infty < 10^{-4}$)
  - **E2E ＆ Preemption (Rule T4, T5)**: `test_e2e_integration.py`（Metal GPU推論、SSE、海馬記憶、プリエンプション）

---

### Sprint 7: 倫理・安全性アライメント & 海馬免疫ガードレール (Phase 2.5) (Days 24〜26)
* [x] **倫理・安全性データセット整備 (`data/`)**:
  - `Anthropic HH-RLHF (Helpful & Harmless)` および `PKU-Alignment / BeaverTails` 日英安全拒絶ペアの自動取得・前処理
  - 攻撃・脱獄・差別・有害コード生成プロンプトに対する建設的拒絶（Constructive Refusal）対話セット構築
* [x] **Layer I 憲法文脈（Constitutional Invariants）注入**:
  - 人権・安全性・透明性のメタ原則を Layer I 大域文脈ベクトル $z_{\text{L1}}$ にエンコードし、Layer V 思考ループの全ステップへトップダウン拘束
* [x] **Phase 2.5: Safety SFT & DPO 学習 (`train/src/train_safety.py`)**:
  - Phase 2 完了モデル重みをベースに、DPO（Direct Preference Optimization）による有害回答の確率抑制最適化
  - 安全な回答（Chosen）と脱獄・有害回答（Rejected）のペアによる嗜好最適化
* [x] **海馬免疫ガードレール (`server/src/memory.rs`)**:
  - ユーザー対話ログから有害プロンプト・データ汚染（Poisoning）を検知し、睡眠学習対象から自動除外する免疫フィルタの実装
  - 基礎倫理アンカー (`anchors.jsonl`) の 30% Replay 混合による長期的倫理規範の忘却防止保証
* [x] **安全性ベンチマーク評価**:
  - Do-Not-Answer / BeaverTails 安全性評価セットでの拒絶率・有害出力ゼロ評価

---

## ⚡ Phase 2: Cotier v1.1 (高速化 & 小脳ショートカット)

### 🛠️ システム実装タスク
* [ ] **小脳型 Fast-Path (0 サイクルバイパス)**:
  - 助詞・句読点・定型構文など確信度が高いトークンは Layer V をバイパスし、1.5〜2倍の高速化（150+ tokens/sec）。
* [ ] **Metal カーネル融合 (MSL: Metal Shading Language)**:
  - Layer V のループ全体を単一の Metal コンピュートシェーダーに融合し、CPU-GPU 間のディスパッチオーバーヘッドを極小化。

### 📚 教育・学習カリキュラム: Phase 3 (小脳バイパス & 濃縮基礎概念オントロジー)
* [ ] **データセット**: 日英 Wikipedia (要約), ConceptNet / Wikidata コアトリプレット, TinyStories (50k サンプル)
* [ ] **① 小脳バイパス蒸留**: PonderNet $k=1$ で解ける平易な構文トークンを識別する小脳確信度ゲートの二値分類学習 ($\mathcal{L}_{\text{bypass}}$)。
* [ ] **② 濃縮基礎知識・世界モデル (Dense World Model Ontology)**:
  - 概念の包含・関係性トリプレット (`Subject-Predicate-Object`) による世界の骨格（物理・地理・CS基礎概念）の獲得。
  - 獲得した概念定義から 1,000 件の基礎知識アンカー (`anchors.jsonl`) を生成し、破滅的忘却を防止。

---

## 🧠 Phase 3: Cotier v1.2 (神経修飾 & 自律制御)

### 🛠️ システム実装タスク
* [ ] **動的 Temperature 制御 (Neuromodulation)**:
  - 予測誤差（Surprise）の大小に応じて、サンプリング温度 $T \in [0.1, 0.9]$ をトークン単位で自動調整。
* [ ] **可塑性（学習率）の自己適応**:
  - ユーザーが強い修正指示を与えた際、一時的に LoRA 学習率を引き上げて即座に吸収。

### 📚 教育・学習カリキュラム: Phase 4 (多段階 MCP 連鎖 & 能動的調査能力)
* [ ] **データセット**: Glaive Function Calling v2, ToolBench, Active-Search-QA, ReAct Synthetics (40k 対話)
* [ ] **① メタ認知 & 知識境界判定 (Metacognition Calibration)**:
  - 「即答できる既知の基礎常識」と「外部調査が必要な未知・詳細事実」を自律判別する対話ペア学習。
* [ ] **② 検索クエリ合成 (Query Formulation)**:
  - ユーザーの曖昧な発話から、的確な検索キーワード・ファイル検索パスを組み立てる能力。
* [ ] **③ ReAct 調査連鎖 (Thought $\to$ `<tool_call>` $\to$ Observation $\to$ Grounded Answer)**:
  - ツール実行結果からノイズを除去し、根拠に基づいた回答（Context Grounding）を生成する多段階対話 SFT。
* [ ] **④ ハルシネーション抑制 DPO (Anti-Hallucination Preference)**:
  - 不確かな推測・知ったかぶり（Rejected） vs 能動的にツールで確認した回答（Chosen）の嗜好最適化。

---

## 🏛️ Phase 4: Cotier v2.0 (高度推論: 視床 & 大脳基底核)

### 🛠️ システム実装タスク
* [ ] **視床ゲート (Thalamic Attention Gating)**:
  - 長文コンテキストの無関係なトークンを動的に間引き、計算量を 50% 削減。
* [ ] **大脳基底核 Go/No-Go 回路**:
  - Layer V で複数の探索仮説（$\tilde{z}_1, \tilde{z}_2, \tilde{z}_3$）を並行生成し、基底核モジュールが最適解を採択（パズル・数学・難解コードでのバックトラック探索）。

### 📚 教育・学習カリキュラム: Phase 5 & 6 (長文視床圧縮 & 理論的思考・基底核強化学習)
* [ ] **Phase 5 (視床長文マスク学習)**:
  - **データ**: Multi-Doc QA, Code Repo Split (20k, $L=2048$)
  - **目的**: クエリに対して重要な Key/Value のみを選択する Sparse Attention マスクの教師あり学習。
* [ ] **Phase 6: 理論的思考 & 大脳基底核 潜在仮説探索 (Theoretical Deductive Reasoning & Basal Ganglia RL)**:
  - **① 合成形式論理 (Formal Logic & Syllogisms)**:
    - 三段論法、命題論理（AND/OR/NOT/IMPLIES）、背理法、矛盾検知（`Logic-Graph Engine` 生成: 50k）
    - 目的: 表層単語の相関に依存せず、Layer V 内部ループで純粋な論理演繹回路を形成。
  - **② 状態遷移 & 実行トレース (State Machine Simulation)**:
    - スタックマシン (`PUSH/POP/ADD/MUL`)、AST走査、変数スコープ追跡（`Micro-Python Tracer` 生成: 30k）
    - 目的: 再帰隠れ状態 $z^{(k)}$ をメモリレジスタとして活用し、逐次状態更新をシミュレート。
  - **③ 因果グラフ & 反実仮想 (Causal DAGs & Counterfactuals)**:
    - DAG依存性解析、「もし A が偽だったら B はどうなるか？」の介入推論 (20k)
    - 目的: トップダウン制約（Layer I $z_{\text{L1}}$）と連動した因果整合性の獲得とハルシネーション抑制。
  - **④ 制約充足 & 自己証明検証 (Constraint Satisfaction & Proof Checking)**:
    - Zebra Puzzle、数独、提示された推論ステップの誤謬検証 (20k)
    - 目的: Layer VI（PonderNet 停止分類器）の停止判定境界の鋭敏化。
  - **⑤ 大脳基底核 Go/No-Go DPO 最適化**:
    - 潜在状態仮説（$\tilde{z}_1, \tilde{z}_2, \tilde{z}_3$）の中から、正解に繋がる推論経路に「Go（高報酬）」を与えるステップレベル DPO / PPO 最適化。

---

## 📊 成功基準 (KPI)

| 指標 | 目標値 | 測定方法 |
| :--- | :--- | :--- |
| **メモリ消費** | **1.5 GB 未満** (KV Cache 込み / Unified Memory) | macOS Activity Monitor / Rust メモリ計測 |
| **推論速度** | **120 〜 150+ tokens/sec** | `cotier bench` (Metal 加速時) |
| **思考スケーリング** | 簡単な語: 1〜2 cycles<br>論理/計算/JSON: 4〜6 cycles | SSE メトリクス (`cortical_metrics.cycles`) |
| **MCP / Tool-Use 精度** | 正しい JSON Schema 引数出力率 **90% 以上** | Glaive Eval セットでのベンチマーク |
| **能動的調査判定精度** | 未知事実に対する自律ツール呼び出し・ハルシネーション抑制率 **90% 以上** | Search-Trigger & Hallucination-Eval |
| **理論的演繹推論精度** | 形式論理演繹・実行トレース正答率 **85% 以上** | Logic-Bench / Python-Trace-Eval |
| **安全性・倫理拒絶率** | 危険・脱獄プロンプトへの安全拒絶率 **95% 以上** | BeaverTails / Do-Not-Answer Eval |
| **破滅的忘却の防止** | 会話成長後も日英SFT・基礎知識精度低下 **5% 未満** | Dolly日英 / Anchor 評価セットでの定期自動評価 |
| **エコシステム互換** | Cursor / Open-WebUI での Tool-Calling 正常動作 | E2E ツール呼び出しテスト通過 |

