# In Silico RaPID Screening

環状ペプチドのドッキングスコア最適化のためのActive Learning / ベイズ最適化パイプライン

---

## 推奨ワークフロー

### インタラクティブモード（推奨）

自動で既存結果を検出し、有効サンプル数を確認:

```bash
python active_learning_from_lead.py --lead "htihswqmhfkin"
```

**実行例:**
```
======================================================================
Active Learning from Lead Sequence - Setup
======================================================================

Lead sequence: htihswqmhfkin
Peptide length: 13

[Step 1] Scanning existing results...
  Scanning: Research_Linux/result
    Found: 104 results
  Scanning: Research_Linux/result_active_learning
    Found: 36 results

  [Combined Statistics]
    Total results: 140
    Outliers (>= 100 or <= -100 kcal/mol): 116
    Usable for training: 24
    Good results (affinity < 0): 18
    Best: htihswqmhfkin (-34.90 kcal/mol)

  [INFO] Need 26 more usable samples to reach target (50)
         Will automatically collect more data during pipeline execution.

[Step 2] Data Collection Strategy

  Target: 50 usable samples (-100 < affinity < 100 kcal/mol)
  Current: 24 usable samples
  Status: Need ~156 more docking runs (estimated outlier rate: 83%)

[Step 3] Active Learning settings
  Iterations: 10
  Batch size: 5
  Docking: N=5, n=100000

[Summary]
  Existing results: 140 total, 24 usable
  Target usable samples: 50
  Estimated initial docking: ~156 runs
  AL iterations: 10 x 5 = 50
  Total estimated docking: ~206

  Note: Pipeline will automatically collect more data if needed
        to reach the target number of usable samples.

Start? [y/N]: y
```

### 自動モード

確認なしでデフォルト値を使用:
```bash
# デフォルト設定で自動実行
python active_learning_from_lead.py --lead "htihswqmhfkin" --no-confirm

# 本番設定で自動実行
python active_learning_from_lead.py --lead "htihswqmhfkin" --production --no-confirm

# クイックテスト（有効サンプル20で開始）
python active_learning_from_lead.py --lead "htihswqmhfkin" --quick --no-confirm
```

---

## 必要ファイル・ディレクトリ

### 必須ファイル

パイプライン実行に必要なファイル一覧:

| ファイル | パス | 説明 | サイズ |
|----------|------|------|--------|
| **1O6K.trg** | `Research_Linux/docking_setup/` | 受容体ターゲットファイル（**最重要**） | ~28MB |
| active_learning_gnn.py | `Research_Linux/Research/` | GNN + GPサロゲートモデル | - |
| transformer_models.py | `Research_Linux/Research/` | 配列エンコーダモデル | - |
| adcp_interface.py | `Research_Linux/Research/` | ADCPラッパー | - |

### ターゲットファイル作成用（1O6K.trgがない場合）

| ファイル | パス | 説明 |
|----------|------|------|
| 1O6K_noligand.pdb | `Research_Linux/Input/` | 受容体PDBファイル |
| Peptideligand.pdbqt | `Research_Linux/Input/` | リガンドPDBQT（結合部位特定用） |

### ディレクトリ構造

```
Research_Linux/
├── Research/               # Pythonモジュール（必須）
│   ├── active_learning_gnn.py
│   ├── transformer_models.py
│   └── adcp_interface.py
│
├── Input/                  # 入力ファイル（.trg作成時に必要）
│   ├── 1O6K_noligand.pdb
│   └── Peptideligand.pdbqt
│
├── docking_setup/          # ターゲットファイル（必須）
│   └── 1O6K.trg
│
├── result/                 # 初期ドッキング結果（自動作成）
├── result_active_learning/ # AL結果（自動作成）
└── al_output/              # 最終出力（自動作成）
```

### 実行前チェック

```bash
# 1. ターゲットファイル確認（最重要）
ls -lh Research_Linux/docking_setup/1O6K.trg

# 2. Pythonモジュール確認
ls Research_Linux/Research/*.py

# 3. 環境確認
micromamba activate in_silico_screening
python -c "import torch; import sklearn; print('OK')"
```

**1O6K.trg がない場合**は、次のクイックスタートの手順3を参照してください。

---

## クイックスタート

### 1. 環境セットアップ（初回のみ）

```bash
# プロジェクトルートで実行
# GNN + Active Learning環境
bash setup_in_silico_screening.sh

# AutoDock CrankPep環境
bash adcpsuite_micromamba.sh
```

### 2. 環境テスト

```bash
micromamba activate in_silico_screening
python test_environment.py
```

### 3. ターゲットファイル作成（初回のみ）

```bash
micromamba activate adcpsuite
cd Research_Linux/docking_setup

# 受容体をPDBQTに変換
agfr -r ../Input/1O6K_noligand.pdb --toPdbqt

# ターゲットファイル作成
agfr -r 1O6K_noligand_rec.pdbqt -l ../Input/Peptideligand.pdbqt -asv 1.1 -o 1O6K

# 確認（約28MBのファイル）
ls -lh 1O6K.trg
```

### 4. Active Learning実行

```bash
# テスト実行
python active_learning_from_lead.py --lead "htihswqmhfkin" --quick --no-confirm

# 本番実行
python active_learning_from_lead.py --lead "htihswqmhfkin" --no-confirm
```

---

## リード配列からの最適化 (active_learning_from_lead.py)

### 主な機能

- **自動データ収集**: 有効データ（外れ値除く）が目標数に達するまで自動でドッキング
- **外れ値フィルタリング**: 結合エネルギー ≥ 100 または ≤ -100 kcal/mol のデータを学習から除外
- **9種類のサロゲートモデル比較**: GIN, GCN, GAT, GraphSAGE, MPNN, GraphTransformer, SeqTransformer, CNN1D, CatBoost
- **自動モデル選択**: 80/20分割でテスト性能を評価し、最良モデルを自動選択
- **リード配列順位表示**: 最終結果でリード配列の順位・改善度を必ず表示
- **パリティプロット**: 各イテレーションで予測 vs 実測を可視化
- **進捗プロット**: 最適化の履歴をグラフ化

### ワークフロー
```
┌─────────────────────────────────────────────────┐
│  1. リード配列を入力                            │
│     例: "htihswqmhfkin"                         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  2. 既存結果を自動スキャン                      │
│     result/ と result_active_learning/ を検索   │
│     外れ値（≥100 or ≤-100 kcal/mol）と有効データを分類│
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  3. 有効データ収集（目標: 50サンプル）          │
│     外れ値を除いた有効データが目標に達するまで  │
│     自動で変異体生成 → ドッキングを繰り返す     │
│     ※初期収集とALで同じドッキング設定を使用    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  4. モデル比較 (80/20 Train/Test Split)         │
│     9種類のモデルでテスト性能を評価             │
│     最良モデルを自動選択（オプション）          │
│     外れ値は学習データから除外済み              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  5. Active Learning（10イテレーション）         │
│     獲得関数で候補選択 → ドッキング             │
│     各イテレーションでパリティプロット作成      │
│     → result_active_learning/に保存             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  6. 最適化された配列を出力 → al_output/         │
│     Top 10候補 + リード配列の順位表示           │
│     リードとの改善度比較                        │
│     最終パリティプロット + 進捗プロット作成     │
└─────────────────────────────────────────────────┘
```

### コマンドライン引数

| 引数 | 短縮 | デフォルト | 説明 |
|------|------|-----------|------|
| `--lead` | `-l` | (必須) | リード配列 |
| `--existing-dir` | | result/ | 既存結果のディレクトリ |
| `--iterations` | `-i` | 10 | ALイテレーション数 |
| `--batch-size` | `-b` | 5 | 各イテレーションの配列数 |
| `--runs` | `-N` | 5 | MC探索回数 |
| `--evals` | `-e` | 100000 | 評価ステップ数 |
| `--timeout` | | 600 | ドッキングタイムアウト（秒） |
| `--acquisition` | | EI | 獲得関数 (EI/UCB/PI) |
| `--min-usable-samples` | | 50 | 学習に必要な最小サンプル数（外れ値除く） |
| `--auto-select-model` | | False | 最良モデルを自動選択 |
| `--production` | | | 本番設定 (timeout=1200) |
| `--quick` | | | クイックテスト (N=1, n=5000, min_usable=20) |
| `--no-confirm` | | | 確認スキップ |

### 使用例

```bash
# インタラクティブモード
python active_learning_from_lead.py --lead "htihswqmhfkin"

# 自動モード
python active_learning_from_lead.py --lead "htihswqmhfkin" --no-confirm

# 有効サンプル100以上を確保してから開始
python active_learning_from_lead.py --lead "htihswqmhfkin" --min-usable-samples 100 --no-confirm

# クイックテスト
python active_learning_from_lead.py --lead "htihswqmhfkin" --quick --no-confirm
```

### 出力ファイル

| ファイル | 説明 |
|----------|------|
| `results_from_lead_*.csv` | 全結果（ランク順、リード配列マーク付き） |
| `final_results_*.json` | サマリー（リード順位・改善度含む） |
| `checkpoint_iterN.json` | 各イテレーションのチェックポイント |
| `parity_plots/` | イテレーション毎のパリティプロット |
| `optimization_progress_*.png` | 最適化進捗プロット |

### 出力フォーマット

**CSV形式:**
```csv
rank,sequence,affinity,is_lead
1,optimized_seq,-38.50,FALSE
2,another_seq,-36.20,FALSE
...
45,htihswqmhfkin,-22.30,TRUE
```

**JSON形式:**
```json
{
  "lead_sequence": "htihswqmhfkin",
  "lead_rank": 45,
  "lead_affinity": -22.30,
  "best_sequence": "optimized_seq",
  "best_affinity": -38.50,
  "improvement": 16.20,
  "total_evaluated": 150,
  ...
}
```

**コンソール出力例:**
```
[Top 10 Sequences]
------------------------------------------------------------
Rank   Sequence                  Affinity (kcal/mol)  Note
------------------------------------------------------------
1      optimized_seq             -38.50
2      another_seq               -36.20
...
------------------------------------------------------------

[Lead Sequence Rank]
  htihswqmhfkin: Rank 45 / 150 (-22.30 kcal/mol)

[Improvement Summary]
  Lead sequence (htihswqmhfkin):
    Affinity: -22.30 kcal/mol
    Rank: 45 / 150
  Best sequence (optimized_seq): -38.50 kcal/mol
  Improvement: 16.20 kcal/mol ✓
```

---

## ドッキング設定

### ADCPコマンド

初期データ収集とActive Learning両方で同じ設定を使用:

```bash
adcp -O -T {receptor}.trg -s {sequence} -o {job_name} \
     -N 5 -n 100000 -cyc -w {output_dir} \
     -nmin 5 -nitr 500 -env implicit -dr -reint
```

### パラメータ詳細

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `-O` | - | 出力を上書き |
| `-T` | {receptor}.trg | ターゲットファイル |
| `-s` | {sequence} | ペプチド配列 |
| `-cyc` | - | 環状ペプチドモード |
| `-N` | 5 | MC探索を5回実行 |
| `-n` | 100000 | 各探索で100,000ステップ評価 |
| `-nmin` | 5 | OpenMM最小化（5回） |
| `-nitr` | 500 | 500イテレーション |
| `-env` | implicit | 暗黙的溶媒環境 |
| `-dr` | - | Dry run optimization |
| `-reint` | - | Re-integration |

### プリセット

| プリセット | 設定 |
|-----------|------|
| デフォルト | N=5, n=100000, timeout=600秒, min_usable=50 |
| `--quick` | N=1, n=5000, timeout=120秒, min_usable=20 |
| `--production` | N=5, n=100000, timeout=1200秒, min_usable=50 |

---

## サロゲートモデル

パイプラインは9種類のサロゲートモデルをサポート:

### グラフベースモデル（分子構造を使用）
| モデル | 説明 |
|--------|------|
| **GIN** | Graph Isomorphism Network（デフォルト） |
| **GCN** | Graph Convolutional Network |
| **GAT** | Graph Attention Network |
| **GraphSAGE** | GraphSAGE |
| **MPNN** | Message Passing Neural Network |
| **GraphTransformer** | Graph Transformer |

### 配列ベースモデル（アミノ酸配列を使用）
| モデル | 説明 |
|--------|------|
| **SeqTransformer** | Sequence Transformer（周期的位置エンコーディング） |
| **CNN1D** | 1D畳み込みネットワーク |
| **CatBoost** | 勾配ブースティング（手動特徴量エンジニアリング） |

---

## モデル比較機能

### 自動モデル選択

`--auto-select-model` オプションを使用すると、初期データでの80/20 Train/Test分割による評価に基づき、最良のモデルを自動選択します。

```bash
python active_learning_from_lead.py --lead "htihswqmhfkin" --auto-select-model
```

評価指標:
- **Test R²**: テストセットでの決定係数（主要指標）
- **Test RMSE**: テストセットでの二乗平均平方根誤差
- **Train R²**: 訓練セットでの決定係数

### 外れ値フィルタリング

結合エネルギーが100 kcal/mol以上、または-100 kcal/mol以下のデータは外れ値として学習データから自動的に除外されます。これにより:
- ドッキング失敗による異常値の影響を排除
- より安定したモデル学習を実現
- 予測精度の向上

### 自動データ収集

有効サンプル（外れ値を除く）が目標数に達するまで、自動的にドッキングを実行:

```
[Phase 1] Collecting training data (target: 50 usable samples)
======================================================================

  [Batch] Need 26 more usable samples, generating 20 variants...
    Generated 20 unique variants
    [1/20] abcdefghijklm: -12.50 kcal/mol
    [2/20] bcdefghijklmn: 156.30 kcal/mol  ← 外れ値
    ...
    Results: 20/20 successful, 8 usable (non-outlier)
    Current usable total: 32/50

  [Batch] Need 18 more usable samples, generating 20 variants...
    ...
```

### パリティプロット

各イテレーションで予測値 vs 実測値のパリティプロットを自動生成:

- 訓練データ（青）とテストデータ（オレンジ）を区別
- R²、RMSE、MAE などの評価指標を表示
- `parity_plots/` ディレクトリに保存

---

## 獲得関数

Active Learningで次に評価する候補を選択:

### Expected Improvement (EI) - 推奨
```
EI(x) = (μ(x) - f_best) · Φ(Z) + σ(x) · φ(Z)
Z = (μ(x) - f_best) / σ(x)
```
- 現在のベストを超える期待改善量
- 探索と活用のバランスが良い

### Upper Confidence Bound (UCB)
```
UCB(x) = μ(x) - β · σ(x)  (最小化の場合)
```
- β で探索の度合いを調整（デフォルト: β=2.0）

### Probability of Improvement (PI)
```
PI(x) = Φ((f_best - μ(x)) / σ(x))
```
- 改善確率のみを考慮（改善量は無視）

---

## 環境構成

### `in_silico_screening` 環境
- Python 3.11
- PyTorch, PyTorch Geometric
- GPyTorch (Gaussian Process)
- RDKit, CatBoost

### `adcpsuite` 環境
- Python 3.7
- AutoDock CrankPep (ADCP)
- OpenMM, ParmEd, OpenBabel

---

## プロジェクト構造

```
In_silico_screening_of_macrocyclic_peptides/
├── active_learning_from_lead.py    # メインスクリプト（推奨）
├── setup_in_silico_screening.sh    # GNN環境セットアップ
├── adcpsuite_micromamba.sh         # ADCP環境セットアップ
├── test_environment.py             # 環境テスト
│
└── Research_Linux/
    ├── Research/                   # Pythonモジュール
    │   ├── active_learning_gnn.py      # GNN + GPパイプライン
    │   ├── transformer_models.py       # Transformer/CNN/CatBoostモデル
    │   └── adcp_interface.py           # ADCPインターフェース
    │
    ├── Input/                      # 入力ファイル
    │   ├── 1O6K_noligand.pdb
    │   └── Peptideligand.pdbqt
    │
    ├── docking_setup/              # ターゲットファイル
    │   └── 1O6K.trg
    │
    ├── result/                     # 初期ドッキング結果
    │   └── {sequence}/             # 各配列のドッキング結果
    │
    ├── result_active_learning/     # AL探索で得た配列の結果
    │   └── {sequence}/             # 機械学習で選択された配列
    │
    └── al_output/                  # Active Learning出力
        ├── results_from_lead_*.csv
        ├── final_results_*.json
        ├── parity_plots/
        └── optimization_progress_*.png
```

---

## コアモジュール詳細 (Research_Linux/Research/)

3つのPythonモジュールが連携して動作します。

### モジュール依存関係

```
active_learning_from_lead.py  (エントリーポイント)
        │
        ▼
active_learning_gnn.py  (ALパイプライン + GNNモデル)
        │
        ├──▶ transformer_models.py  (配列ベースモデル)
        │
        └──▶ adcp_interface.py      (ドッキング実行)
```

### 1. adcp_interface.py - ADCPドッキングインターフェース

AutoDock CrankPep (ADCP) のPythonラッパー。

**主要クラス:**

| クラス | 説明 |
|--------|------|
| `DockingResult` | ドッキング結果を格納するデータクラス |
| `ADCPRunner` | ドッキング実行を管理 |
| `ADCPActiveLearningIntegration` | ALパイプラインとADCPを統合 |

**ADCPRunner の主要メソッド:**
```python
runner = ADCPRunner(receptor_file="1O6K.trg", work_dir="result/")

# 単一配列のドッキング
result = runner.run_docking("HTIHSWQMHFKIN")
print(result.affinity)  # -34.90 kcal/mol

# バッチドッキング
results = runner.batch_docking(["HTIHSWQMHFKIN", "ACDEFGHIKLMN"])
```

**内部処理:**
1. ADCPコマンドを構築 (`_build_adcp_command`)
2. subprocessでADCP実行
3. .dlgファイルをパース (`_parse_dlg_file`)
4. `DockingResult`オブジェクトを返却

### 2. transformer_models.py - 配列エンコーダモデル

アミノ酸配列を埋め込みベクトルに変換するモデル群。

**主要クラス:**

| クラス | 入力 | 出力 | 説明 |
|--------|------|------|------|
| `SequenceTransformerEncoder` | 配列 | 埋め込み | BERT風Transformer + 周期的位置エンコーディング |
| `GraphTransformerEncoder` | グラフ | 埋め込み | グラフ構造にTransformerを適用 |
| `CNN1DEncoder` | 配列 | 埋め込み | マルチスケール1D畳み込み (kernel: 3,5,7) |
| `CatBoostSurrogate` | 配列 | 予測値+不確実性 | 勾配ブースティング + Virtual Ensembles |

**補助クラス/関数:**
- `CyclicPositionalEncoding`: 環状ペプチド用の周期的位置エンコーディング
- `AA_VOCAB`: アミノ酸→トークンID辞書
- `AA_PROPERTIES`: アミノ酸の物理化学的特性（疎水性、分子量、電荷、極性、芳香族性）
- `AA_SECONDARY_STRUCTURE`: 二次構造傾向パラメータ（α-helix, β-sheet, Turn）

**使用例:**
```python
from transformer_models import SequenceTransformerEncoder, CatBoostSurrogate

# Transformerエンコーダ
encoder = SequenceTransformerEncoder(d_model=128, out_channels=64)
embeddings = encoder(["HTIHSWQMHFKIN", "ACDEFGHIKLMN"])  # [2, 64]

# CatBoostサロゲート
catboost = CatBoostSurrogate()
catboost.fit(sequences, y_values)
mu, std = catboost.predict(new_sequences)  # 予測値と不確実性
```

### 3. active_learning_gnn.py - メインパイプライン

Active Learningのコアロジックを実装。

**主要クラス:**

| クラス | 説明 |
|--------|------|
| `Config` | パイプライン設定（パス、ハイパーパラメータ等） |
| `CustomGNN` | スクラッチ実装のGNN (GIN/GCN/GAT/GraphSAGE/MPNN) |
| `GPModel` | Gaussian Processモデル (GPyTorch) |
| `SurrogateModel` | GNN/Transformer + GP サロゲートモデル |
| `ALState` | Active Learning状態管理 |
| `ActiveLearningPipeline` | メインパイプライン |

**GNNレイヤー（スクラッチ実装）:**
- `GINConvLayer`: Graph Isomorphism Network
- `GraphConvLayer`: Graph Convolutional Network
- `GATConvLayer`: Graph Attention Network
- `GraphSAGEConvLayer`: GraphSAGE
- `MPNNConvLayer`: Message Passing Neural Network

**処理フロー:**
```
1. sequence_to_graph()
   └── RDKitでアミノ酸配列を分子グラフに変換
   └── 環状化処理（N末端-C末端結合）
   └── ノード特徴量: [原子番号, 次数, 電荷, 混成, 芳香族性, ラジカル電子]

2. CustomGNN.forward()
   └── グラフ畳み込み層 × num_layers
   └── グローバル平均プーリング
   └── 出力: [batch_size, out_channels]

3. GPModel
   └── RBFカーネル + ARD
   └── 予測: 平均μ + 標準偏差σ

4. 獲得関数
   └── expected_improvement(μ, σ, best_y)
   └── upper_confidence_bound(μ, σ, β)
   └── probability_of_improvement(μ, σ, best_y)

5. ActiveLearningPipeline.run_iteration()
   └── サロゲートモデル学習
   └── 候補選択（獲得関数）
   └── ドッキング実行
   └── モデル更新
```

**使用例:**
```python
from active_learning_gnn import Config, ActiveLearningPipeline

config = Config(
    n_iterations=10,
    batch_size=5,
    acquisition="EI"
)

pipeline = ActiveLearningPipeline(config)
history = pipeline.run()

# 上位候補取得
top_candidates = pipeline.get_top(10)
```

---

### 結果ディレクトリの使い分け
| ディレクトリ | 内容 | ドッキング設定 |
|-------------|------|---------------|
| `Research_Linux/result/` | 初期データ収集 | N=5, n=100000, implicit |
| `Research_Linux/result_active_learning/` | AL結果 | N=5, n=100000, implicit（同じ設定） |
| `Research_Linux/al_output/` | 最終結果 | - |

---

## トラブルシューティング

### 環境が作成できない
```bash
micromamba env remove -n in_silico_screening
bash setup_in_silico_screening.sh
```

### パッケージのインポートエラー
```bash
micromamba activate in_silico_screening
python test_environment.py
```

### ターゲットファイルが見つからない
```bash
cd Research_Linux/docking_setup
micromamba activate adcpsuite
agfr -r ../Input/1O6K_noligand.pdb --toPdbqt
agfr -r 1O6K_noligand_rec.pdbqt -l ../Input/Peptideligand.pdbqt -asv 1.1 -o 1O6K
```

### Affinityが正の値（外れ値）になる
- ドッキングが収束していない可能性
- デフォルト設定（N=5, n=100000）で改善されるはず
- 良い結果の目安: -25 〜 -35 kcal/mol

### 有効サンプルが集まらない
- 外れ値率が高い場合、自動的に追加ドッキングが実行される
- `--min-usable-samples` を下げることも可能（ただし精度低下の恐れ）

---

## 進捗確認コマンド

```bash
# ドッキング済み配列数
ls Research_Linux/result/ | wc -l
ls Research_Linux/result_active_learning/ | wc -l

# 現在のドッキング状況
ps aux | grep adcp | grep -v grep

# 最新の結果確認
ls -lt Research_Linux/al_output/ | head -10
```

---

## 参考

- AutoDock CrankPep: https://ccsb.scripps.edu/adcp/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- GPyTorch: https://gpytorch.ai/
- RDKit: https://www.rdkit.org/

---

**更新日**: 2026-01-31 (v4.4 - リード配列順位表示機能追加)
