#!/usr/bin/env python3
"""
Active Learning from Lead Sequence - 環状ペプチド最適化パイプライン
====================================================================

【概要】
このスクリプトは、リード配列（初期のペプチド配列）から出発して、
機械学習を使ってより結合力の強いペプチド配列を自動的に探索します。

【Active Learningとは？】
- 少ないデータで効率的に最適解を見つける機械学習手法
- サロゲートモデル（代理モデル）で結合力を予測
- 獲得関数で「次に試すべき配列」を賢く選択
- 実際にドッキングして結果を確認 → モデルを更新 → 繰り返し

【ワークフロー】
1. リード配列を入力
   例: "htihswqmhfkin"（13アミノ酸の環状ペプチド）

2. 既存のドッキング結果をスキャン
   過去に計算した結果があれば再利用（時間節約）

3. 初期データ収集
   リード配列の変異体を生成 → ドッキング → 学習データを蓄積
   ※外れ値（-100〜100 kcal/mol外）は除外

4. サロゲートモデルの学習
   9種類のモデルから最適なものを自動選択
   - グラフベース: GIN, GCN, GAT, GraphSAGE, MPNN, GraphTransformer
   - 配列ベース: SeqTransformer, CNN1D, CatBoost

5. Active Learning最適化ループ
   予測 → 候補選択 → ドッキング → 更新 を繰り返す

6. 結果出力
   最適化された配列のランキング + リード配列との比較

【使用例】
    # インタラクティブモード（推奨）
    python active_learning_from_lead.py --lead "htihswqmhfkin"

    # クイックテスト（動作確認用、数分で完了）
    python active_learning_from_lead.py --lead "htihswqmhfkin" --quick --no-confirm

    # 本番実行（高精度、数時間かかる）
    python active_learning_from_lead.py --lead "htihswqmhfkin" --production --no-confirm

【必要ファイル】
    - Research_Linux/docking_setup/1O6K.trg（受容体ターゲットファイル）
    - Research_Linux/Research/*.py（モデル実装）

【出力ファイル】
    - al_output/results_from_lead_*.csv: 全配列のランキング
    - al_output/final_results_*.json: サマリー（リード順位含む）
    - al_output/parity_plots/: 予測精度の可視化
"""

import os
import sys
import random
import argparse
import json
import subprocess
import re
from datetime import datetime
from pathlib import Path

# パスの設定
SCRIPT_DIR = Path(__file__).parent.absolute()
RESEARCH_LINUX_DIR = SCRIPT_DIR / "Research_Linux"
RESEARCH_DIR = RESEARCH_LINUX_DIR / "Research"  # Pythonモジュール用
sys.path.insert(0, str(RESEARCH_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# =============================================================================
# 定数定義
# =============================================================================

# 20種類の天然アミノ酸（小文字）
# タンパク質を構成する標準的なアミノ酸。ペプチド配列はこれらの組み合わせで表現される
# 例: "htihswqmhfkin" は H(ヒスチジン), T(スレオニン), I(イソロイシン)... の配列
AMINO_ACIDS = list("acdefghiklmnpqrstvwy")

# 利用可能なモデルタイプ
# GNN（グラフニューラルネットワーク）系:
#   - GIN: Graph Isomorphism Network（グラフの構造を識別する能力が高い）
#   - GCN: Graph Convolutional Network（基本的なグラフ畳み込み）
#   - GAT: Graph Attention Network（重要なノードに注目する機構付き）
#   - GraphSAGE: サンプリングベースのGNN（大規模グラフ向け）
#   - MPNN: Message Passing Neural Network（汎用的なメッセージパッシング）
#   - GraphTransformer: Transformerアーキテクチャをグラフに適用
# 配列ベース:
#   - SeqTransformer: 配列をTransformerで処理
#   - CNN1D: 1次元畳み込み（配列のローカルパターンを学習）
#   - CatBoost: 勾配ブースティング（アミノ酸特徴量を直接使用）
MODEL_TYPES = ["GIN", "GCN", "GAT", "GraphSAGE", "MPNN",
               "SeqTransformer", "GraphTransformer", "CNN1D", "CatBoost"]

# 外れ値の閾値（この範囲外のaffinityは学習から除外）
# ドッキングスコアが異常に高い/低い値は計算エラーの可能性があるため除外
# 通常のドッキングスコアは -20 〜 0 kcal/mol 程度
AFFINITY_OUTLIER_THRESHOLD_HIGH = 100.0   # >= 100 kcal/mol は外れ値（計算エラー）
AFFINITY_OUTLIER_THRESHOLD_LOW = -100.0   # <= -100 kcal/mol は外れ値（計算エラー）
AFFINITY_OUTLIER_THRESHOLD = AFFINITY_OUTLIER_THRESHOLD_HIGH  # 後方互換性のため維持

# モデル学習に必要な最小サンプル数（外れ値を除いた数）
# 機械学習モデルが意味のある予測をするためには、十分なデータが必要
# 50サンプル未満だと過学習のリスクが高くなる
MIN_USABLE_SAMPLES = 50


def filter_outliers(sequences, affinities,
                    threshold_high=AFFINITY_OUTLIER_THRESHOLD_HIGH,
                    threshold_low=AFFINITY_OUTLIER_THRESHOLD_LOW):
    """
    外れ値（高すぎる/低すぎるaffinity）を除外する

    【なぜ外れ値を除外するのか？】
    ドッキング計算では、まれに異常な値（計算エラー）が出ることがあります。
    これらをそのまま学習に使うと、モデルの予測精度が悪化します。

    【処理の流れ】
    1. すべてのaffinity値をチェック
    2. 閾値の範囲外（-100以下 or 100以上）のデータを除外
    3. 有効なデータのみを返す

    Parameters
    ----------
    sequences : list
        ペプチド配列のリスト（例: ["htihswqm...", "acdefghi..."]）
    affinities : np.ndarray
        各配列に対応するAffinity値（ドッキングスコア、単位: kcal/mol）
        負の値ほど結合力が強い
    threshold_high : float
        上限閾値。この値以上は外れ値として除外（デフォルト: 100）
    threshold_low : float
        下限閾値。この値以下は外れ値として除外（デフォルト: -100）

    Returns
    -------
    tuple (filtered_sequences, filtered_affinities, n_removed)
        - filtered_sequences: 外れ値を除いた配列リスト
        - filtered_affinities: 外れ値を除いたaffinity配列
        - n_removed: 除外されたサンプル数
    """
    affinities = np.array(affinities)
    # 範囲内のデータのみ保持: threshold_low < affinity < threshold_high
    # 例: -100 < affinity < 100 のデータのみ残す
    mask = (affinities > threshold_low) & (affinities < threshold_high)
    filtered_seqs = [s for s, m in zip(sequences, mask) if m]
    filtered_affs = affinities[mask]
    n_removed = len(sequences) - len(filtered_seqs)
    return filtered_seqs, filtered_affs, n_removed


# =============================================================================
# 既存結果の読み込み
# =============================================================================
# ドッキング計算は時間がかかるため、一度計算した結果は再利用します。
# このセクションでは、過去に計算した結果をファイルから読み込みます。

def scan_existing_results(result_dir: Path) -> dict:
    """
    既存のドッキング結果をスキャンして読み込む

    【この関数の役割】
    過去に実行したドッキング計算の結果を読み込み、再計算を避けます。
    1つの配列につき数分〜数十分かかるため、再利用は大幅な時間節約になります。

    【ディレクトリ構造】
    result_dir/
    ├── htihswqmhfkin/           # 配列名のディレクトリ
    │   └── result_htihswqmhfkin_summary.dlg  # ドッキング結果ファイル
    ├── acdefghiklmn/
    │   └── result_acdefghiklmn_summary.dlg
    └── ...

    Args:
        result_dir: 結果ディレクトリへのパス

    Returns:
        {sequence: affinity} の辞書
        例: {"htihswqmhfkin": -8.5, "acdefghiklmn": -6.2, ...}
    """
    results = {}

    if not result_dir.exists():
        return results

    for seq_dir in result_dir.iterdir():
        if not seq_dir.is_dir():
            continue

        sequence = seq_dir.name.lower()

        # 配列名の検証（アミノ酸のみで構成されているか）
        if not all(c in AMINO_ACIDS for c in sequence):
            continue

        # DLGファイルを探す
        dlg_files = list(seq_dir.glob("*_summary.dlg"))
        if not dlg_files:
            continue

        dlg_file = dlg_files[0]

        try:
            affinity = parse_dlg_affinity(dlg_file)
            if affinity is not None:
                results[sequence] = affinity
        except Exception:
            continue

    return results


def parse_dlg_affinity(dlg_file: Path) -> float:
    """DLGファイルからAffinityを抽出"""
    with open(dlg_file, 'r') as f:
        content = f.read()

    # OpenMMリスコアリング結果を優先
    omm_match = re.search(
        r'OMM Ranking:\s+1\s+1\s+\d+\s+([-\d.]+)',
        content
    )
    if omm_match:
        return float(omm_match.group(1))

    # 通常のaffinity（mode 1）
    affinity_match = re.search(
        r'^\s+1\s+([-\d.]+)\s+',
        content,
        re.MULTILINE
    )
    if affinity_match:
        return float(affinity_match.group(1))

    return None


def load_csv_results(result_dir: Path) -> dict:
    """
    CSVファイルから結果を読み込む（従来形式との互換性）。
    """
    results = {}

    # result_dir内のCSVファイルを探す
    for csv_file in result_dir.glob("*.csv"):
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            # ヘッダーをスキップ
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    sequence = parts[0].lower().strip()
                    try:
                        affinity = float(parts[1])
                        if sequence and all(c in AMINO_ACIDS for c in sequence):
                            results[sequence] = affinity
                    except ValueError:
                        continue
        except Exception:
            continue

    return results


# =============================================================================
# 変異体生成
# =============================================================================
# リード配列（最初に与えられた配列）を変異させて、新しい候補配列を生成します。
#
# 【変異とは？】
# 配列中のアミノ酸を別のアミノ酸に置き換えること。
# 例: "HTIH..." → "ATIH..."（1番目のHをAに変異）
#
# 【なぜ変異体を生成するのか？】
# リード配列に近い配列は、似たような性質（結合力）を持つ可能性が高い。
# 少しずつ変異させることで、効率的に良い配列を探索できる。

def random_mutate_all_positions(peptide: str) -> str:
    """
    全位置を元と異なるアミノ酸にランダム変異させる

    【用途】
    リード配列とは大きく異なる配列を生成したい場合に使用。
    探索空間の多様性を確保するため。

    【例】
    入力: "htih" → 出力: "acde"（全位置が変更される）

    Args:
        peptide: 元のペプチド配列（例: "htihswqmhfkin"）

    Returns:
        変異後の配列（全位置が元と異なる）
    """
    mutated = []
    for aa in peptide.lower():
        # 元と同じアミノ酸を除外した候補
        choices = [x for x in AMINO_ACIDS if x != aa]
        mutated.append(random.choice(choices))
    return "".join(mutated)


def random_mutate_n_positions(peptide: str, n_mutations: int = None) -> str:
    """
    ランダムなn個の位置を変異させる。

    Args:
        peptide: 元のペプチド配列
        n_mutations: 変異させる位置数（Noneの場合はランダム）

    Returns:
        変異後の配列
    """
    peptide = peptide.lower()
    length = len(peptide)

    if n_mutations is None:
        # 1〜全位置のランダムな数
        n_mutations = random.randint(1, length)

    n_mutations = min(n_mutations, length)

    # 変異させる位置をランダムに選択
    positions = random.sample(range(length), n_mutations)

    mutated = list(peptide)
    for pos in positions:
        original_aa = mutated[pos]
        choices = [x for x in AMINO_ACIDS if x != original_aa]
        mutated[pos] = random.choice(choices)

    return "".join(mutated)


def generate_mutation_library(
    lead: str,
    n_variants: int = 100,
    mutation_strategy: str = "mixed",
    include_lead: bool = True
) -> list:
    """
    リード配列から変異体ライブラリを生成。

    Args:
        lead: リード配列
        n_variants: 生成する変異体数
        mutation_strategy: 変異戦略
            - "all": 全位置変異
            - "single": 1点変異のみ
            - "mixed": 混合（1点〜全位置）
        include_lead: リード配列自体も含めるか

    Returns:
        変異体配列のリスト
    """
    lead = lead.lower()
    library = set()

    # リード配列を含める場合
    if include_lead:
        library.add(lead)

    # 変異体を生成
    attempts = 0
    max_attempts = n_variants * 10  # 無限ループ防止

    while len(library) < n_variants and attempts < max_attempts:
        attempts += 1

        if mutation_strategy == "all":
            # 全位置変異
            variant = random_mutate_all_positions(lead)
        elif mutation_strategy == "single":
            # 1点変異
            variant = random_mutate_n_positions(lead, n_mutations=1)
        else:  # mixed
            # 1点〜全位置のランダム
            variant = random_mutate_n_positions(lead, n_mutations=None)

        library.add(variant)

    return list(library)


def generate_diverse_library(
    lead: str,
    n_variants: int = 100,
    include_lead: bool = True
) -> list:
    """
    多様性を確保した変異体ライブラリを生成。

    - 1点変異: 20%
    - 2-3点変異: 30%
    - 4-6点変異: 30%
    - 全位置変異: 20%

    Args:
        lead: リード配列
        n_variants: 生成する変異体数
        include_lead: リード配列自体も含めるか

    Returns:
        変異体配列のリスト
    """
    lead = lead.lower()
    length = len(lead)
    library = set()

    if include_lead:
        library.add(lead)

    # 各カテゴリの目標数
    n_single = int(n_variants * 0.2)
    n_few = int(n_variants * 0.3)
    n_medium = int(n_variants * 0.3)
    n_all = n_variants - n_single - n_few - n_medium

    # 1点変異
    for _ in range(n_single * 2):
        if len([x for x in library if x != lead]) >= n_single:
            break
        variant = random_mutate_n_positions(lead, n_mutations=1)
        library.add(variant)

    # 2-3点変異
    for _ in range(n_few * 2):
        if len(library) >= n_single + n_few + (1 if include_lead else 0):
            break
        n_mut = random.choice([2, 3])
        variant = random_mutate_n_positions(lead, n_mutations=min(n_mut, length))
        library.add(variant)

    # 4-6点変異
    for _ in range(n_medium * 2):
        if len(library) >= n_single + n_few + n_medium + (1 if include_lead else 0):
            break
        n_mut = random.choice([4, 5, 6])
        variant = random_mutate_n_positions(lead, n_mutations=min(n_mut, length))
        library.add(variant)

    # 全位置変異で残りを埋める
    while len(library) < n_variants:
        variant = random_mutate_all_positions(lead)
        library.add(variant)

    return list(library)[:n_variants]


# =============================================================================
# ドッキング実行
# =============================================================================
# AutoDock CrankPep (ADCP) を使って、ペプチドと受容体タンパク質の
# ドッキングシミュレーションを実行します。
#
# 【ドッキングとは？】
# ペプチド（薬の候補）がタンパク質（標的）にどのように結合するかを
# コンピュータでシミュレーションすること。
# 結合力（affinity）はkcal/mol単位で表され、負の値が大きいほど強く結合。
# 例: -10 kcal/mol は -5 kcal/mol より強い結合

class DockingRunner:
    """
    ADCPドッキングを実行するクラス

    【このクラスの役割】
    1. ペプチド配列を受け取る
    2. ADCPコマンドを構築して実行
    3. 結果ファイル（.dlg）からaffinityを抽出

    【使用するツール】
    - AutoDock CrankPep (ADCP): 環状ペプチド専用のドッキングツール
    - OpenMM: より精密なエネルギー計算（リスコアリング）
    """

    def __init__(
        self,
        receptor_file: str,
        work_dir: str,
        n_runs: int = 5,
        n_evals: int = 100000,
        timeout: int = 600,
        use_omm: bool = True,
        verbose: bool = False,
        use_implicit_env: bool = True,
        nmin: int = 5,
        nitr: int = 500
    ):
        self.receptor_file = receptor_file
        self.work_dir = Path(work_dir)
        self.n_runs = n_runs
        self.n_evals = n_evals
        self.timeout = timeout
        self.use_omm = use_omm
        self.verbose = verbose
        self.use_implicit_env = use_implicit_env
        self.nmin = nmin
        self.nitr = nitr

        # 作業ディレクトリ作成
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_single_docking(self, sequence: str) -> dict:
        """単一配列のドッキングを実行"""
        sequence = sequence.lower()
        job_name = f"result_{sequence}"
        output_dir = self.work_dir / sequence
        output_dir.mkdir(parents=True, exist_ok=True)

        # ADCPコマンド構築
        # 参考: adcp -O -T 3Q47.trg -cyc -N 5 -n 100000 -nmin 5 -nitr 500 -env implicit -dr -reint
        cmd = [
            "micromamba", "run", "-n", "adcpsuite",
            "adcp",
            "-O",
            "-T", str(self.receptor_file),
            "-s", sequence,
            "-o", job_name,
            "-N", str(self.n_runs),
            "-n", str(self.n_evals),
            "-cyc",
            "-w", str(output_dir),
            "-nmin", str(self.nmin),
            "-nitr", str(self.nitr),
            "-dr",
            "-reint"
        ]

        # 暗黙的溶媒環境を使用
        if self.use_implicit_env:
            cmd.extend(["-env", "implicit"])

        if self.verbose:
            print(f"    Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # DLGファイルがあれば結果をパース（ADCPがエラー終了してもドッキング自体は成功している場合がある）
            dlg_file = output_dir / f"{job_name}_summary.dlg"
            if dlg_file.exists():
                try:
                    affinity = self._parse_result(output_dir, sequence)
                    return {
                        "sequence": sequence,
                        "success": True,
                        "affinity": affinity
                    }
                except (ValueError, FileNotFoundError):
                    pass

            if result.returncode != 0:
                return {
                    "sequence": sequence,
                    "success": False,
                    "error": f"ADCP failed: {result.stderr[:200]}"
                }

            # 結果をパース
            affinity = self._parse_result(output_dir, sequence)

            return {
                "sequence": sequence,
                "success": True,
                "affinity": affinity
            }

        except subprocess.TimeoutExpired:
            return {
                "sequence": sequence,
                "success": False,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "sequence": sequence,
                "success": False,
                "error": str(e)
            }

    def _parse_result(self, output_dir: Path, sequence: str) -> float:
        """ドッキング結果からAffinityを抽出"""
        dlg_file = output_dir / f"result_{sequence}_summary.dlg"

        if not dlg_file.exists():
            raise FileNotFoundError(f"DLG file not found: {dlg_file}")

        with open(dlg_file, 'r') as f:
            content = f.read()

        # OpenMMリスコアリング結果を優先
        omm_match = re.search(
            r'OMM Ranking:\s+1\s+1\s+\d+\s+([-\d.]+)',
            content
        )
        if omm_match:
            return float(omm_match.group(1))

        # 通常のaffinity
        affinity_match = re.search(
            r'^\s+1\s+([-\d.]+)\s+',
            content,
            re.MULTILINE
        )
        if affinity_match:
            return float(affinity_match.group(1))

        raise ValueError(f"Could not parse affinity from {dlg_file}")

    def run_batch_docking(self, sequences: list, progress_callback=None) -> list:
        """複数配列のドッキングを実行"""
        results = []

        for i, seq in enumerate(sequences):
            if progress_callback:
                progress_callback(i, len(sequences), seq)

            result = self.run_single_docking(seq)
            results.append(result)

            if result["success"]:
                print(f"    [{i+1}/{len(sequences)}] {seq}: {result['affinity']:.2f} kcal/mol")
            else:
                print(f"    [{i+1}/{len(sequences)}] {seq}: FAILED - {result.get('error', 'Unknown')}")

        return results


# =============================================================================
# モデル比較・評価機能
# =============================================================================
# 機械学習モデルの性能を評価し、最適なモデルを選択するための機能です。
#
# 【評価指標】
# - R² (決定係数): 1に近いほど予測精度が高い（1.0が完璧）
# - RMSE (平均二乗誤差の平方根): 小さいほど誤差が少ない
# - MAE (平均絶対誤差): 予測と実測の差の平均
#
# 【Train/Test分割】
# データを訓練用(80%)とテスト用(20%)に分け、
# 訓練データで学習し、未知のテストデータで性能を評価します。
# テストデータでの性能が本当の予測能力を示します。

def create_parity_plot(y_true_train, y_pred_train, y_true_test, y_pred_test,
                       y_std_train=None, y_std_test=None,
                       output_path=None, iteration=None):
    """
    パリティプロット（予測値 vs 実測値）を作成して保存

    【パリティプロットとは？】
    X軸に実測値、Y軸に予測値をプロットしたグラフ。
    理想的には全点が対角線（y=x）上に乗る。
    対角線から離れているほど予測精度が悪い。

    【プロットの見方】
    - 青い点: 訓練データ（モデルが学習に使ったデータ）
    - 赤い四角: テストデータ（モデルが見ていない未知のデータ）
    - 点線: 理想線（y=x、完璧な予測）
    - テストデータ（赤）が対角線近くにあれば、未知データへの予測も良好
    """
    y_true_train = np.array(y_true_train)
    y_pred_train = np.array(y_pred_train)
    y_true_test = np.array(y_true_test)
    y_pred_test = np.array(y_pred_test)

    # Test setの評価指標の計算
    r2_test = r2_score(y_true_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    mae_test = mean_absolute_error(y_true_test, y_pred_test)

    # Train setの評価指標の計算
    r2_train = r2_score(y_true_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
    mae_train = mean_absolute_error(y_true_train, y_pred_train)

    # プロット作成
    fig, ax = plt.subplots(figsize=(8, 8))

    # Trainデータポイント（青）
    ax.scatter(y_true_train, y_pred_train, alpha=0.5, s=40, color='blue',
              label=f'Train (n={len(y_true_train)})')

    # Testデータポイント（赤）
    ax.scatter(y_true_test, y_pred_test, alpha=0.8, s=60, color='red',
              marker='s', label=f'Test (n={len(y_true_test)})')

    # 理想線（y=x）
    all_true = np.concatenate([y_true_train, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_test])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    margin = (max_val - min_val) * 0.1
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, 'k--', linewidth=2, label='Ideal (y=x)')

    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_xlabel('Actual Affinity (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted Affinity (kcal/mol)', fontsize=12)

    if iteration is not None:
        title = f'Parity Plot - Iteration {iteration}'
    else:
        title = 'Parity Plot - Surrogate Model'
    ax.set_title(title, fontsize=14)

    # 評価指標をテキストボックスで表示
    textstr = (f'Train: R²={r2_train:.3f}, RMSE={rmse_train:.2f}\n'
               f'Test:  R²={r2_test:.3f}, RMSE={rmse_test:.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'    Parity plot saved: {output_path}')

    plt.close()

    return {
        'r2_train': r2_train, 'rmse_train': rmse_train, 'mae_train': mae_train, 'n_train': len(y_true_train),
        'r2_test': r2_test, 'rmse_test': rmse_test, 'mae_test': mae_test, 'n_test': len(y_true_test)
    }


def plot_optimization_progress(iteration_history, output_path=None):
    """
    Active Learning最適化進捗プロット
    """
    if not iteration_history:
        return

    iters = [h['iteration'] for h in iteration_history]
    best_affs = [h['best_aff'] for h in iteration_history]
    n_seqs = [h['n_sequences'] for h in iteration_history]

    # Test R²があれば取得
    r2_test = [h.get('r2_test') for h in iteration_history]
    has_r2 = any(r2 is not None for r2 in r2_test)

    # プロット作成
    if has_r2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Best Affinity
    axes[0].plot(iters, best_affs, 'b-o', lw=2, markersize=6)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Affinity (kcal/mol)')
    axes[0].set_title('Optimization Progress')
    axes[0].grid(True, alpha=0.3)

    # Number of Sequences
    axes[1].plot(iters, n_seqs, 'g-s', lw=2, markersize=6)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Total Sequences')
    axes[1].set_title('Data Accumulation')
    axes[1].grid(True, alpha=0.3)

    # Test R² (if available)
    if has_r2:
        valid_r2 = [(i, r) for i, r in zip(iters, r2_test) if r is not None]
        if valid_r2:
            r2_iters, r2_vals = zip(*valid_r2)
            axes[2].plot(r2_iters, r2_vals, 'r-^', lw=2, markersize=6)
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Test R²')
            axes[2].set_title('Model Performance')
            axes[2].set_ylim([0, 1])
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Progress plot saved: {output_path}')

    plt.close()


def compare_models(sequences, affinities, config, output_dir):
    """
    複数のモデルアーキテクチャを比較
    """
    from active_learning_gnn import CustomGNN, SurrogateModel
    from transformer_models import (SequenceTransformerEncoder, GraphTransformerEncoder,
                                     CNN1DEncoder, CatBoostSurrogate)

    print(f'\n{"="*70}')
    print('Comparing Model Architectures')
    print(f'{"="*70}')

    # 外れ値を除外
    sequences_filtered, affinities_filtered, n_removed = filter_outliers(sequences, affinities)
    if n_removed > 0:
        print(f'\nFiltered {n_removed} outliers (affinity >= {AFFINITY_OUTLIER_THRESHOLD_HIGH} or <= {AFFINITY_OUTLIER_THRESHOLD_LOW})')
        print(f'Using {len(sequences_filtered)} samples for model comparison')

    # Train/Test分割
    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences_filtered, affinities_filtered,
        test_size=0.2,
        random_state=config.seed if hasattr(config, 'seed') else 42
    )
    print(f'\nData split: Train={len(seq_train)}, Test={len(seq_test)}')

    results = {}
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for idx, model_type in enumerate(MODEL_TYPES):
        print(f'\n[{idx+1}/{len(MODEL_TYPES)}] Training {model_type}...')

        try:
            # モデル作成
            if model_type == "SeqTransformer":
                encoder = SequenceTransformerEncoder(
                    d_model=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    num_layers=config.gnn_num_layers,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="sequence")
            elif model_type == "GraphTransformer":
                encoder = GraphTransformerEncoder(
                    in_channels=6,
                    hidden_channels=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    num_layers=config.gnn_num_layers,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="graph")
            elif model_type == "CNN1D":
                encoder = CNN1DEncoder(
                    embed_dim=config.gnn_hidden_dim // 2,
                    num_filters=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="sequence")
            elif model_type == "CatBoost":
                surrogate = CatBoostSurrogate(
                    n_estimators=500,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=config.seed if hasattr(config, 'seed') else 42,
                    verbose=False
                )
            else:
                # GNNモデル
                encoder = CustomGNN(
                    in_channels=6,
                    hidden_channels=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    num_layers=config.gnn_num_layers,
                    dropout=config.gnn_dropout,
                    conv_type=model_type
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="graph")

            # 学習
            surrogate.fit(seq_train, y_train, verbose=False)

            # 予測
            y_pred_train, y_std_train = surrogate.predict(seq_train)
            y_pred_test, y_std_test = surrogate.predict(seq_test)

            # 有効な予測のみ
            valid_train = ~np.isnan(y_pred_train) & ~np.isnan(y_train)
            valid_test = ~np.isnan(y_pred_test) & ~np.isnan(y_test)

            if valid_train.sum() > 0 and valid_test.sum() > 0:
                r2_train = r2_score(y_train[valid_train], y_pred_train[valid_train])
                r2_test = r2_score(y_test[valid_test], y_pred_test[valid_test])
                rmse_train = np.sqrt(mean_squared_error(y_train[valid_train], y_pred_train[valid_train]))
                rmse_test = np.sqrt(mean_squared_error(y_test[valid_test], y_pred_test[valid_test]))

                results[model_type] = {
                    'r2_train': r2_train,
                    'r2_test': r2_test,
                    'rmse_train': rmse_train,
                    'rmse_test': rmse_test,
                    'surrogate': surrogate
                }

                print(f'    Train R²={r2_train:.4f}, Test R²={r2_test:.4f}')
                print(f'    Train RMSE={rmse_train:.2f}, Test RMSE={rmse_test:.2f}')

                # サブプロット作成
                ax = axes[idx]
                ax.scatter(y_train[valid_train], y_pred_train[valid_train],
                          alpha=0.5, s=30, color='blue', label=f'Train')
                ax.scatter(y_test[valid_test], y_pred_test[valid_test],
                          alpha=0.8, s=50, color='red', marker='s', label=f'Test')

                all_vals = np.concatenate([y_train[valid_train], y_test[valid_test],
                                          y_pred_train[valid_train], y_pred_test[valid_test]])
                min_val, max_val = all_vals.min(), all_vals.max()
                margin = (max_val - min_val) * 0.1
                line_range = [min_val - margin, max_val + margin]
                ax.plot(line_range, line_range, 'k--', linewidth=1.5)

                ax.set_xlim(line_range)
                ax.set_ylim(line_range)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{model_type}\nTrain R²={r2_train:.3f}, Test R²={r2_test:.3f}')
                ax.legend(loc='lower right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            else:
                results[model_type] = {'error': 'No valid predictions'}
                print(f'    ✗ No valid predictions')
                axes[idx].text(0.5, 0.5, f'{model_type}\nNo valid predictions',
                              ha='center', va='center', transform=axes[idx].transAxes)

        except Exception as e:
            results[model_type] = {'error': str(e)}
            print(f'    ✗ Error: {e}')
            axes[idx].text(0.5, 0.5, f'{model_type}\nError',
                          ha='center', va='center', transform=axes[idx].transAxes)

    # サマリー（10番目のサブプロット）
    ax = axes[9]
    ax.axis('off')

    summary_text = "Model Comparison Summary\n" + "="*35 + "\n\n"
    summary_text += f"{'Type':<15} {'Train R²':>10} {'Test R²':>10}\n"
    summary_text += "-"*37 + "\n"

    valid_results = {k: v for k, v in results.items() if 'r2_test' in v}
    for model_type in MODEL_TYPES:
        if model_type in valid_results:
            r = valid_results[model_type]
            summary_text += f"{model_type:<15} {r['r2_train']:>10.4f} {r['r2_test']:>10.4f}\n"
        else:
            summary_text += f"{model_type:<15} {'N/A':>10} {'N/A':>10}\n"

    if valid_results:
        best_type = max(valid_results.keys(), key=lambda x: valid_results[x]['r2_test'])
        summary_text += "\n" + "="*37 + "\n"
        summary_text += f"Best (Test R²): {best_type}"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Model Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    comparison_path = Path(output_dir) / 'parity_plots' / 'model_comparison.png'
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f'\nComparison plot saved: {comparison_path}')
    plt.close()

    return results


def select_best_model(results, auto_select=False):
    """
    モデル選択（自動 or ユーザー選択）
    """
    valid_results = {k: v for k, v in results.items() if 'r2_test' in v}

    if not valid_results:
        print("No valid models available. Using default GIN.")
        return "GIN"

    # Test R²でソート
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['r2_test'], reverse=True)

    print(f'\n{"="*70}')
    print('Model Selection')
    print(f'{"="*70}')
    print(f'\n{"No.":<4} {"Type":<16} {"Train R²":>10} {"Test R²":>10} {"Test RMSE":>12}')
    print("-"*55)

    for i, (model_type, r) in enumerate(sorted_results, 1):
        marker = " <-- Best" if i == 1 else ""
        print(f'{i:<4} {model_type:<16} {r["r2_train"]:>10.4f} {r["r2_test"]:>10.4f} {r["rmse_test"]:>12.2f}{marker}')

    if auto_select:
        selected = sorted_results[0][0]
        print(f'\nAuto-selected: {selected} (best Test R²)')
        return selected

    print("-"*55)
    print(f'0    Auto (use best: {sorted_results[0][0]})')

    while True:
        try:
            choice = input(f'\nSelect model (0-{len(sorted_results)}): ').strip()
            if choice == '' or choice == '0':
                selected = sorted_results[0][0]
                print(f'Auto-selected: {selected}')
                return selected
            choice = int(choice)
            if 1 <= choice <= len(sorted_results):
                selected = sorted_results[choice-1][0]
                print(f'Selected: {selected}')
                return selected
            print(f'Please enter 0-{len(sorted_results)}')
        except ValueError:
            print('Please enter a valid number')
        except KeyboardInterrupt:
            print('\nUsing best model')
            return sorted_results[0][0]


def create_surrogate_model(model_type, config):
    """
    指定されたタイプのサロゲートモデルを作成
    """
    from active_learning_gnn import CustomGNN, SurrogateModel
    from transformer_models import (SequenceTransformerEncoder, GraphTransformerEncoder,
                                     CNN1DEncoder, CatBoostSurrogate)

    if model_type == "SeqTransformer":
        encoder = SequenceTransformerEncoder(
            d_model=config.gnn_hidden_dim,
            out_channels=config.gnn_output_dim,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout
        )
        return SurrogateModel(encoder, config, encoder_mode="sequence")
    elif model_type == "GraphTransformer":
        encoder = GraphTransformerEncoder(
            in_channels=6,
            hidden_channels=config.gnn_hidden_dim,
            out_channels=config.gnn_output_dim,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout
        )
        return SurrogateModel(encoder, config, encoder_mode="graph")
    elif model_type == "CNN1D":
        encoder = CNN1DEncoder(
            embed_dim=config.gnn_hidden_dim // 2,
            num_filters=config.gnn_hidden_dim,
            out_channels=config.gnn_output_dim,
            dropout=config.gnn_dropout
        )
        return SurrogateModel(encoder, config, encoder_mode="sequence")
    elif model_type == "CatBoost":
        return CatBoostSurrogate(
            n_estimators=500,
            learning_rate=0.1,
            depth=6,
            random_seed=config.seed if hasattr(config, 'seed') else 42,
            verbose=False
        )
    else:
        # GNNモデル（GIN, GCN, GAT, GraphSAGE, MPNN）
        encoder = CustomGNN(
            in_channels=6,
            hidden_channels=config.gnn_hidden_dim,
            out_channels=config.gnn_output_dim,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            conv_type=model_type
        )
        return SurrogateModel(encoder, config, encoder_mode="graph")


# =============================================================================
# メインパイプライン
# =============================================================================
# Active Learning（能動学習）の本体。
# 少ないデータで効率的に最適な配列を見つけるための反復的な最適化を行います。
#
# 【Active Learningの仕組み】
# 1. 現在のデータでモデルを学習
# 2. モデルで全候補の結合力を予測
# 3. 「有望そう」かつ「不確実性が高い」候補を選択（獲得関数）
# 4. 選択した候補をドッキング計算で評価
# 5. 結果をデータに追加して1に戻る
#
# 【なぜActive Learningが有効？】
# - ドッキング計算は1配列あたり数分〜数十分かかる
# - 全配列を計算するのは非現実的（13アミノ酸なら20^13 = 約10^17通り）
# - 賢く選んだ少数の配列だけを計算し、効率的に最適解に近づく

class ActiveLearningFromLead:
    """
    リード配列からActive Learningを開始するパイプライン

    【このクラスの役割】
    1. リード配列から変異体ライブラリを生成
    2. 初期データを収集（ドッキング計算）
    3. モデルを学習・選択
    4. Active Learningループで最適化
    5. 結果を保存・可視化

    【使用例】
    pipeline = ActiveLearningFromLead(
        lead_sequence="htihswqmhfkin",
        receptor_file="1O6K.trg",
        ...
    )
    pipeline.run()
    """

    def __init__(
        self,
        lead_sequence: str,
        receptor_file: str,
        output_dir: str,
        result_dir: str,
        al_result_dir: str,
        existing_result_dir: str = None,
        n_init: int = 100,
        n_iterations: int = 10,
        batch_size: int = 5,
        n_runs: int = 1,
        n_evals: int = 5000,
        timeout: int = 60,
        use_omm: bool = True,
        acquisition: str = "EI",
        seed: int = 42,
        verbose: bool = False,
        use_existing: bool = False,
        min_existing: int = 100,
        preloaded_results: dict = None,
        auto_select_model: bool = False,
        min_usable_samples: int = MIN_USABLE_SAMPLES
    ):
        self.lead_sequence = lead_sequence.lower()
        self.receptor_file = receptor_file
        self.output_dir = Path(output_dir)
        self.result_dir = Path(result_dir)  # 初期ドッキング結果
        self.al_result_dir = Path(al_result_dir)  # AL結果
        self.existing_result_dir = Path(existing_result_dir) if existing_result_dir else None

        self.n_init = n_init  # 追加で生成するサンプル数
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.n_runs = n_runs
        self.n_evals = n_evals
        self.timeout = timeout
        self.use_omm = use_omm
        self.acquisition = acquisition
        self.seed = seed
        self.verbose = verbose
        self.use_existing = use_existing
        self.min_existing = min_existing
        self.auto_select_model = auto_select_model
        self.min_usable_samples = min_usable_samples

        # 乱数シード設定
        random.seed(seed)
        np.random.seed(seed)

        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.al_result_dir.mkdir(parents=True, exist_ok=True)

        # 結果を保存（事前ロード結果があれば使用）
        self.all_results = preloaded_results.copy() if preloaded_results else {}

        # 共通のドッキング設定
        docking_params = {
            "receptor_file": receptor_file,
            "n_runs": n_runs,
            "n_evals": n_evals,
            "timeout": timeout,
            "use_omm": use_omm,
            "verbose": verbose,
            "use_implicit_env": True,  # 暗黙的溶媒環境
            "nmin": 5,                 # OpenMM最小化
            "nitr": 500                # イテレーション数
        }

        # ドッキングランナー（初期用）
        self.docking_runner = DockingRunner(
            work_dir=result_dir,  # result/に保存
            **docking_params
        )

        # ドッキングランナー（AL用）- 同じ設定
        self.al_docking_runner = DockingRunner(
            work_dir=al_result_dir,  # result_active_learning/に保存
            **docking_params
        )

    def _count_usable_samples(self):
        """外れ値を除いた有効なサンプル数をカウント"""
        affinities = np.array(list(self.all_results.values()))
        # 範囲内のデータのみカウント: -100 < affinity < 100
        n_usable = np.sum((affinities > AFFINITY_OUTLIER_THRESHOLD_LOW) &
                          (affinities < AFFINITY_OUTLIER_THRESHOLD_HIGH))
        return int(n_usable)

    def _print_data_status(self):
        """現在のデータ状況を表示"""
        affinities = np.array(list(self.all_results.values()))
        n_total = len(affinities)
        n_outliers_high = np.sum(affinities >= AFFINITY_OUTLIER_THRESHOLD_HIGH)
        n_outliers_low = np.sum(affinities <= AFFINITY_OUTLIER_THRESHOLD_LOW)
        n_outliers = n_outliers_high + n_outliers_low
        n_usable = n_total - n_outliers
        n_good = np.sum(affinities < 0)

        print(f"  Total results: {n_total}")
        print(f"  Outliers (>= {AFFINITY_OUTLIER_THRESHOLD_HIGH} or <= {AFFINITY_OUTLIER_THRESHOLD_LOW}): {n_outliers}")
        print(f"  Usable for training: {n_usable}")
        print(f"  Good results (< 0): {n_good}")

        if n_total > 0:
            best_seq = min(self.all_results, key=self.all_results.get)
            print(f"  Best: {best_seq} ({min(affinities):.2f} kcal/mol)")

        return n_usable

    def run(self):
        """パイプライン全体を実行"""
        print("\n" + "=" * 70)
        print("Starting Active Learning Pipeline")
        print("=" * 70)

        # 現在の状況を表示
        print(f"\n[Initial Status]")
        n_usable = self._print_data_status()

        # Phase 1: 外れ値を除いた有効データがmin_usable_samples以上になるまでドッキング
        target = self.min_usable_samples
        print(f"\n[Phase 1] Collecting training data (target: {target} usable samples)")
        print("=" * 70)

        batch_size = 20  # 一度に生成・ドッキングする数
        max_attempts = 500  # 最大試行回数（無限ループ防止）
        total_docked = 0

        while n_usable < target and total_docked < max_attempts:
            n_needed = target - n_usable
            # 外れ値率を考慮して多めに生成（仮に50%が外れ値と仮定）
            n_to_generate = min(batch_size, max(n_needed * 3, 10))

            print(f"\n  [Batch] Need {n_needed} more usable samples, generating {n_to_generate} variants...")

            # 既存配列を除外して新しい変異体を生成
            existing_seqs = set(self.all_results.keys())
            library = generate_diverse_library(
                self.lead_sequence,
                n_variants=n_to_generate + len(existing_seqs) + 100,
                include_lead=True
            )
            library = [seq for seq in library if seq not in existing_seqs][:n_to_generate]

            if not library:
                print("  WARNING: Could not generate more unique variants")
                break

            print(f"    Generated {len(library)} unique variants")

            # ドッキング実行
            docking_results = self.docking_runner.run_batch_docking(library)

            # 結果を保存
            n_success = 0
            n_usable_new = 0
            for res in docking_results:
                if res["success"]:
                    self.all_results[res["sequence"]] = res["affinity"]
                    n_success += 1
                    # 外れ値でないかチェック（-100 < affinity < 100）
                    aff = res["affinity"]
                    if aff > AFFINITY_OUTLIER_THRESHOLD_LOW and aff < AFFINITY_OUTLIER_THRESHOLD_HIGH:
                        n_usable_new += 1

            total_docked += len(library)
            n_usable = self._count_usable_samples()

            print(f"    Results: {n_success}/{len(library)} successful, {n_usable_new} usable (non-outlier)")
            print(f"    Current usable total: {n_usable}/{target}")

        if n_usable < target:
            print(f"\n  WARNING: Could only collect {n_usable} usable samples (target was {target})")
            print(f"           Proceeding with available data...")

        # 最終データ状況
        print(f"\n[Data Collection Complete]")
        self._print_data_status()

        if len(self.all_results) == 0:
            print("ERROR: No docking results available. Aborting.")
            return

        # Phase 2: Active Learning
        print(f"\n[Phase 2] Starting Active Learning ({self.n_iterations} iterations)...")
        print("=" * 70)

        self._run_active_learning(auto_select_model=getattr(self, 'auto_select_model', False))

        # Final results
        self._save_final_results()

        # 完了メッセージ
        print("\n" + "=" * 70)
        print("Active Learning Pipeline Completed!")
        print("=" * 70)

    def _run_active_learning(self, auto_select_model=False):
        """Active Learningループを実行（モデル比較機能付き）"""
        try:
            from active_learning_gnn import (
                Config, CustomGNN, SurrogateModel,
                generate_random_sequences, generate_mutants,
                expected_improvement, upper_confidence_bound, probability_of_improvement
            )
            import torch
        except ImportError as e:
            print(f"Import error: {e}")
            print("Make sure you're in the correct environment.")
            return

        # Config設定
        config = Config()
        config.acquisition = self.acquisition
        config.batch_size = self.batch_size
        config.seed = self.seed
        config.peptide_length = len(self.lead_sequence)

        # 現在のデータ
        sequences = list(self.all_results.keys())
        affinities = np.array([self.all_results[s] for s in sequences])

        # --- Phase 3a: モデル比較・選択 ---
        selected_model_type = "GIN"  # デフォルト
        parity_dir = self.output_dir / "parity_plots"
        parity_dir.mkdir(parents=True, exist_ok=True)

        if len(sequences) >= 20:
            print(f"\n[Phase 3a] Comparing model architectures...")
            model_results = compare_models(
                sequences, affinities, config, str(self.output_dir)
            )
            selected_model_type = select_best_model(model_results, auto_select=auto_select_model)
        else:
            print(f"\n[Phase 3a] Not enough data for model comparison (n={len(sequences)}). Using GIN.")

        # 選択されたモデルを保存（最終結果保存時に使用）
        self.selected_model_type = selected_model_type

        print(f"\n[Phase 3b] Starting Active Learning with {selected_model_type}...")

        # イテレーション履歴
        iteration_history = []

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n{'='*70}")
            print(f"--- Iteration {iteration}/{self.n_iterations} ---")
            print(f"{'='*70}")

            # イテレーション毎の変数初期化
            iter_metrics = {}

            # 現在のデータを取得
            all_sequences = list(self.all_results.keys())
            all_affinities = np.array([self.all_results[s] for s in all_sequences])

            # 外れ値を除外して学習用データを作成
            sequences, affinities, n_outliers = filter_outliers(all_sequences, all_affinities)
            affinities = np.array(affinities)

            # ベストでソート
            sorted_idx = np.argsort(affinities)
            sequences = [sequences[i] for i in sorted_idx]
            affinities = affinities[sorted_idx]

            print(f"  Total data: {len(all_sequences)} sequences")
            if n_outliers > 0:
                print(f"  Outliers removed: {n_outliers} (outside {AFFINITY_OUTLIER_THRESHOLD_LOW} ~ {AFFINITY_OUTLIER_THRESHOLD_HIGH})")
            print(f"  Training data: {len(sequences)} sequences")
            print(f"  Current best: {affinities[0]:.2f} kcal/mol ({sequences[0]})")

            # 選択されたモデルでサロゲートモデルを作成
            surrogate = create_surrogate_model(selected_model_type, config)

            # --- パリティプロット作成（Train/Test分割） ---
            if len(sequences) >= 10:
                print(f"  Creating parity plot (80/20 split)...")
                try:
                    seq_train, seq_test, y_train, y_test = train_test_split(
                        sequences, affinities,
                        test_size=0.2,
                        random_state=self.seed
                    )

                    # Trainデータでモデルを学習
                    surrogate.fit(seq_train, y_train, verbose=False)

                    # 予測
                    y_pred_train, _ = surrogate.predict(seq_train)
                    y_pred_test, _ = surrogate.predict(seq_test)

                    # 有効な予測のみ
                    valid_train = ~np.isnan(y_pred_train) & ~np.isnan(y_train)
                    valid_test = ~np.isnan(y_pred_test) & ~np.isnan(y_test)

                    if valid_train.sum() > 0 and valid_test.sum() > 0:
                        plot_path = parity_dir / f'parity_iter_{iteration:03d}.png'
                        metrics = create_parity_plot(
                            y_train[valid_train], y_pred_train[valid_train],
                            y_test[valid_test], y_pred_test[valid_test],
                            output_path=str(plot_path),
                            iteration=iteration
                        )
                        print(f"    Train R²={metrics['r2_train']:.4f}, Test R²={metrics['r2_test']:.4f}")

                        # メトリクスを一時保存
                        iter_metrics = {
                            'r2_train': metrics['r2_train'],
                            'r2_test': metrics['r2_test'],
                            'rmse_test': metrics['rmse_test']
                        }
                except Exception as e:
                    print(f"    Parity plot failed: {e}")
                    iter_metrics = {}

            # 全データでモデルを再学習（候補選択用）
            surrogate = create_surrogate_model(selected_model_type, config)
            surrogate.fit(sequences, affinities, verbose=False)

            # 候補生成
            existing = set(sequences)

            # ランダム候補
            random_cands = generate_random_sequences(
                n=200,
                length=len(self.lead_sequence),
                existing=existing
            )

            # 上位配列からの変異体
            top_seqs = sequences[:min(10, len(sequences))]
            mutant_cands = []
            for seq in top_seqs:
                mutants = generate_mutants(seq, n_mut=2, n_var=20, existing=existing)
                mutant_cands.extend(mutants)
                existing.update(mutants)

            candidates = list(set(random_cands + mutant_cands))
            print(f"  Generated {len(candidates)} candidates")

            # 予測
            mu, std = surrogate.predict(candidates)

            if len(mu) == 0:
                print("  Warning: No valid predictions, skipping iteration")
                continue

            # 獲得関数で選択
            best_y = affinities[0]
            if self.acquisition == "EI":
                scores = expected_improvement(mu, std, best_y)
            elif self.acquisition == "UCB":
                scores = upper_confidence_bound(mu, std, beta=2.0)
            elif self.acquisition == "PI":
                scores = probability_of_improvement(mu, std, best_y)
            else:
                scores = expected_improvement(mu, std, best_y)

            # 上位選択
            top_idx = np.argsort(scores)[-self.batch_size:][::-1]
            selected = [candidates[i] for i in top_idx]

            # 予測値を記録（後で実測値と比較）
            predicted_affinities = {candidates[i]: mu[i] for i in top_idx}

            print(f"  Selected {len(selected)} candidates for docking")
            print(f"  Predicted affinities: {[f'{mu[i]:.1f}' for i in top_idx]}")

            # ドッキング実行
            docking_results = self.al_docking_runner.run_batch_docking(selected)

            # 結果を追加 & 予測 vs 実測を比較
            new_best = False
            current_best = affinities[0]
            prediction_errors = []

            for res in docking_results:
                if res["success"]:
                    self.all_results[res["sequence"]] = res["affinity"]
                    actual = res["affinity"]
                    predicted = predicted_affinities.get(res["sequence"], None)

                    if predicted is not None:
                        error = actual - predicted
                        prediction_errors.append(error)
                        print(f"    {res['sequence']}: Predicted={predicted:.1f}, Actual={actual:.1f}, Error={error:.1f}")

                    if actual < current_best:
                        new_best = True
                        current_best = actual

            if prediction_errors:
                mae = np.mean(np.abs(prediction_errors))
                print(f"  Prediction MAE this iteration: {mae:.2f} kcal/mol")

            if new_best:
                best_seq = min(self.all_results, key=self.all_results.get)
                print(f"  *** New best found: {current_best:.2f} kcal/mol ({best_seq}) ***")

            # イテレーション履歴に記録
            current_best_seq = min(self.all_results, key=self.all_results.get)
            current_best_aff = self.all_results[current_best_seq]

            iter_record = {
                'iteration': iteration,
                'n_sequences': len(self.all_results),
                'best_seq': current_best_seq,
                'best_aff': float(current_best_aff),
                'new_sequences': [res["sequence"] for res in docking_results if res["success"]],
                'new_affinities': [float(res["affinity"]) for res in docking_results if res["success"]],
                'prediction_mae': float(np.mean(np.abs(prediction_errors))) if prediction_errors else None
            }
            # パリティプロットのメトリクスがあれば追加
            if 'iter_metrics' in dir() and iter_metrics:
                iter_record.update(iter_metrics)
            iteration_history.append(iter_record)

            # 中間保存
            self._save_checkpoint(iteration)

        # イテレーション履歴を保存
        self.iteration_history = iteration_history  # 最終結果表示用に保存
        if iteration_history:
            history_file = self.output_dir / "iteration_history.json"
            with open(history_file, 'w') as f:
                json.dump(iteration_history, f, indent=2)
            print(f"\nIteration history saved: {history_file}")

            # イテレーション履歴のサマリー表示
            print(f"\n[Iteration History Summary]")
            print("-" * 70)
            print(f"{'Iter':<6} {'Seqs':<8} {'Best Aff':<12} {'Test R²':<10} {'Pred MAE':<10}")
            print("-" * 70)
            for h in iteration_history:
                r2 = f"{h.get('r2_test', 'N/A'):.4f}" if h.get('r2_test') is not None else 'N/A'
                mae = f"{h.get('prediction_mae', 'N/A'):.2f}" if h.get('prediction_mae') is not None else 'N/A'
                print(f"{h['iteration']:<6} {h['n_sequences']:<8} {h['best_aff']:<12.2f} {r2:<10} {mae:<10}")

    def _save_checkpoint(self, iteration: int):
        """チェックポイントを保存"""
        checkpoint = {
            "iteration": iteration,
            "lead_sequence": self.lead_sequence,
            "n_results": len(self.all_results),
            "best_sequence": min(self.all_results, key=self.all_results.get),
            "best_affinity": min(self.all_results.values()),
            "results": self.all_results
        }

        checkpoint_file = self.output_dir / f"checkpoint_iter{iteration}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _save_final_results(self):
        """最終結果を保存（パリティプロット含む）"""
        from active_learning_gnn import Config

        print("\n" + "=" * 70)
        print("Creating Final Parity Plot")
        print("=" * 70)

        # 最終パリティプロット作成
        sequences = list(self.all_results.keys())
        affinities = np.array([self.all_results[s] for s in sequences])

        # 外れ値を除外（学習データとして不適切な値）
        sequences, affinities, n_outliers = filter_outliers(sequences, affinities)
        if n_outliers > 0:
            print(f"Excluded {n_outliers} outliers (outside {AFFINITY_OUTLIER_THRESHOLD_LOW} ~ {AFFINITY_OUTLIER_THRESHOLD_HIGH} kcal/mol) from final parity plot")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if len(sequences) >= 10:
            try:
                config = Config()
                config.seed = self.seed

                # Train/Test分割
                seq_train, seq_test, y_train, y_test = train_test_split(
                    sequences, affinities,
                    test_size=0.2,
                    random_state=self.seed
                )

                # 最終モデルを作成・学習
                surrogate = create_surrogate_model(
                    getattr(self, 'selected_model_type', 'GIN'), config
                )
                surrogate.fit(seq_train, y_train, verbose=False)

                # 予測
                y_pred_train, _ = surrogate.predict(seq_train)
                y_pred_test, _ = surrogate.predict(seq_test)

                # 有効な予測のみ
                valid_train = ~np.isnan(y_pred_train) & ~np.isnan(y_train)
                valid_test = ~np.isnan(y_pred_test) & ~np.isnan(y_test)

                if valid_train.sum() > 0 and valid_test.sum() > 0:
                    parity_dir = self.output_dir / "parity_plots"
                    parity_dir.mkdir(parents=True, exist_ok=True)
                    final_plot_path = parity_dir / f'parity_final_{timestamp}.png'

                    metrics = create_parity_plot(
                        y_train[valid_train], y_pred_train[valid_train],
                        y_test[valid_test], y_pred_test[valid_test],
                        output_path=str(final_plot_path),
                        iteration=None
                    )

                    print(f'\nFinal Model Metrics:')
                    print(f'  Train: n={metrics["n_train"]}, R²={metrics["r2_train"]:.4f}, RMSE={metrics["rmse_train"]:.2f}')
                    print(f'  Test:  n={metrics["n_test"]}, R²={metrics["r2_test"]:.4f}, RMSE={metrics["rmse_test"]:.2f}')
            except Exception as e:
                print(f'Final parity plot failed: {e}')
        else:
            print(f'Not enough data for final parity plot (n={len(sequences)})')

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        # ソート
        sorted_results = sorted(self.all_results.items(), key=lambda x: x[1])

        print(f"\n[Overall Statistics]")
        print(f"  Total iterations: {self.n_iterations}")
        print(f"  Total sequences evaluated: {len(self.all_results)}")
        print(f"  Best: {sorted_results[0][0]} ({sorted_results[0][1]:.2f} kcal/mol)")

        # リードの順位を先に計算
        lead_rank = None
        lead_affinity = None
        for i, (seq, aff) in enumerate(sorted_results, 1):
            if seq == self.lead_sequence:
                lead_rank = i
                lead_affinity = aff
                break

        print(f"\n[Top 10 Sequences]")
        print("-" * 60)
        print(f"{'Rank':<6} {'Sequence':<25} {'Affinity (kcal/mol)':<20} {'Note':<8}")
        print("-" * 60)

        for i, (seq, aff) in enumerate(sorted_results[:10], 1):
            note = "<- LEAD" if seq == self.lead_sequence else ""
            print(f"{i:<6} {seq:<25} {aff:<20.2f} {note}")

        # リードがTop 10に入っていない場合、別途表示
        if lead_rank and lead_rank > 10:
            print("...")
            print(f"{lead_rank:<6} {self.lead_sequence:<25} {lead_affinity:<20.2f} <- LEAD")
        print("-" * 60)

        # リードの順位を必ず表示
        print(f"\n[Lead Sequence Rank]")
        if lead_rank:
            print(f"  {self.lead_sequence}: Rank {lead_rank} / {len(sorted_results)} ({lead_affinity:.2f} kcal/mol)")
        else:
            print(f"  {self.lead_sequence}: NOT FOUND in results")

        # CSVで保存
        csv_file = self.output_dir / f"results_from_lead_{timestamp}.csv"

        with open(csv_file, 'w') as f:
            f.write("rank,sequence,affinity,is_lead\n")
            for i, (seq, aff) in enumerate(sorted_results, 1):
                is_lead = "TRUE" if seq == self.lead_sequence else "FALSE"
                f.write(f"{i},{seq},{aff},{is_lead}\n")

        print(f"\nResults saved to: {csv_file}")

        # リード情報（既に計算済み）
        best_affinity = sorted_results[0][1]
        improvement = (lead_affinity - best_affinity) if lead_affinity else 0

        # JSON保存（リード情報を含む）
        json_file = self.output_dir / f"final_results_{timestamp}.json"
        final_data = {
            "lead_sequence": self.lead_sequence,
            "lead_rank": lead_rank,
            "lead_affinity": lead_affinity,
            "n_init": self.n_init,
            "n_iterations": self.n_iterations,
            "batch_size": self.batch_size,
            "total_evaluated": len(self.all_results),
            "best_sequence": sorted_results[0][0],
            "best_affinity": sorted_results[0][1],
            "improvement": improvement,
            "top_10": [{"sequence": s, "affinity": a} for s, a in sorted_results[:10]],
            "model_type": getattr(self, 'selected_model_type', 'GIN')
        }

        with open(json_file, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"Final data saved to: {json_file}")

        # リードとの比較（Improvement Summary）
        print(f"\n[Improvement Summary]")
        if lead_rank:
            print(f"  Lead sequence ({self.lead_sequence}):")
            print(f"    Affinity: {lead_affinity:.2f} kcal/mol")
            print(f"    Rank: {lead_rank} / {len(sorted_results)}")
            print(f"  Best sequence ({sorted_results[0][0]}): {best_affinity:.2f} kcal/mol")
            if improvement > 0:
                print(f"  Improvement: {improvement:.2f} kcal/mol ✓")
            elif improvement == 0:
                print(f"  Lead sequence is the best!")
            else:
                print(f"  No improvement from lead sequence")
        else:
            print(f"  [Warning] Lead sequence ({self.lead_sequence}) not found in results")

        # 進捗プロット作成
        if hasattr(self, 'iteration_history') and self.iteration_history:
            progress_plot_path = self.output_dir / 'parity_plots' / f'progress_{timestamp}.png'
            plot_optimization_progress(self.iteration_history, str(progress_plot_path))

        # 出力ファイル一覧
        print(f"\n[Output Files]")
        print(f"  Results CSV: {csv_file}")
        print(f"  Results JSON: {json_file}")
        print(f"  Parity plots: {self.output_dir / 'parity_plots'}/")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning from Lead Sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python active_learning_from_lead.py --lead "carsrtyriyqrp" --quick --no-confirm

  # Custom initial library size
  python active_learning_from_lead.py --lead "carsrtyriyqrp" --init-size 50 -i 5 -b 3

  # Production run
  python active_learning_from_lead.py --lead "carsrtyriyqrp" --production
        """
    )

    # Required
    parser.add_argument(
        "--lead", "-l",
        required=True,
        help="Lead peptide sequence (e.g., 'carsrtyriyqrp')"
    )

    # Initial library
    parser.add_argument(
        "--init-size", "-n",
        type=int,
        default=100,
        help="Initial library size (default: 100)"
    )

    # Active Learning
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
        help="Number of AL iterations (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Sequences per iteration (default: 5)"
    )
    parser.add_argument(
        "--acquisition",
        choices=["EI", "UCB", "PI"],
        default="EI",
        help="Acquisition function (default: EI)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Docking
    parser.add_argument(
        "--runs", "-N",
        type=int,
        default=5,
        help="MC search runs (default: 5)"
    )
    parser.add_argument(
        "--evals", "-e",
        type=int,
        default=100000,
        help="Evaluation steps per run (default: 100000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Docking timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--no-omm-rescoring",
        action="store_true",
        help="Disable OpenMM rescoring"
    )

    # Paths (相対パス: Research_Linux/ 以下)
    parser.add_argument(
        "--receptor",
        default=str(RESEARCH_LINUX_DIR / "docking_setup" / "1O6K.trg"),
        help="Receptor target file (.trg)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESEARCH_LINUX_DIR / "al_output"),
        help="Output directory for CSV/JSON results"
    )
    parser.add_argument(
        "--result-dir",
        default=str(RESEARCH_LINUX_DIR / "result"),
        help="Directory for initial docking results"
    )
    parser.add_argument(
        "--al-result-dir",
        default=str(RESEARCH_LINUX_DIR / "result_active_learning"),
        help="Directory for Active Learning docking results"
    )
    parser.add_argument(
        "--existing-dir",
        default=str(RESEARCH_LINUX_DIR / "result"),
        help="Directory with existing docking results to scan"
    )

    # Presets
    parser.add_argument(
        "--production",
        action="store_true",
        help="Production preset: init=100, N=5, n=100000, i=10, b=5"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test preset: init=10, N=1, n=1000, i=2, b=2"
    )

    # Model Selection
    parser.add_argument(
        "--auto-select-model",
        action="store_true",
        help="Automatically select best model (skip interactive selection)"
    )
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default=None,
        help="Force use of specific model (skip comparison)"
    )

    # Data collection
    parser.add_argument(
        "--min-usable-samples",
        type=int,
        default=MIN_USABLE_SAMPLES,
        help=f"Minimum usable (non-outlier) samples for training (default: {MIN_USABLE_SAMPLES})"
    )

    # Other
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def interactive_setup(args) -> dict:
    """
    インタラクティブに設定を行う。

    Returns:
        設定の辞書（n_additional, skip_initial_docking など）
    """
    existing_dir = Path(args.existing_dir)
    al_result_dir = Path(args.al_result_dir)

    print("=" * 70)
    print("Active Learning from Lead Sequence - Setup")
    print("=" * 70)
    print(f"\nLead sequence: {args.lead}")
    print(f"Peptide length: {len(args.lead)}")
    print()

    # 既存結果をスキャン（両方のディレクトリをスキャン）
    print("[Step 1] Scanning existing results...")
    print(f"  Scanning: {existing_dir}")
    existing_results = scan_existing_results(existing_dir)
    n_from_result = len(existing_results)
    print(f"    Found: {n_from_result} results")

    # AL結果ディレクトリもスキャン
    print(f"  Scanning: {al_result_dir}")
    al_results = scan_existing_results(al_result_dir)
    n_from_al = len(al_results)
    print(f"    Found: {n_from_al} results")

    # 結果をマージ（重複は後者が優先）
    existing_results.update(al_results)
    n_existing = len(existing_results)

    if n_existing > 0:
        affinities = list(existing_results.values())
        best_seq = min(existing_results, key=existing_results.get)
        best_aff = existing_results[best_seq]

        # 統計情報
        n_good = sum(1 for a in affinities if a < 0)
        n_outliers_high = sum(1 for a in affinities if a >= AFFINITY_OUTLIER_THRESHOLD_HIGH)
        n_outliers_low = sum(1 for a in affinities if a <= AFFINITY_OUTLIER_THRESHOLD_LOW)
        n_outliers = n_outliers_high + n_outliers_low
        n_usable = n_existing - n_outliers

        print()
        print(f"  [Combined Statistics]")
        print(f"    Total results: {n_existing}")
        print(f"    Outliers (>= {AFFINITY_OUTLIER_THRESHOLD_HIGH} or <= {AFFINITY_OUTLIER_THRESHOLD_LOW}): {n_outliers}")
        print(f"    Usable for training: {n_usable}")
        print(f"    Good results (affinity < 0): {n_good}")
        print(f"    Best: {best_seq} ({best_aff:.2f} kcal/mol)")
        print(f"    Mean (all): {np.mean(affinities):.2f} kcal/mol")

        min_target = args.min_usable_samples
        if n_usable < min_target:
            n_needed = min_target - n_usable
            print()
            print(f"  [INFO] Need {n_needed} more usable samples to reach target ({min_target})")
            print(f"         Will automatically collect more data during pipeline execution.")
    else:
        n_usable = 0
        n_outliers = 0
        min_target = args.min_usable_samples
        print("  No existing results found.")
        print(f"  Will collect {min_target} usable samples during pipeline execution.")

    print()

    # Step 2: データ収集の説明
    print("[Step 2] Data Collection Strategy")
    print()
    print(f"  Target: {min_target} usable samples ({AFFINITY_OUTLIER_THRESHOLD_LOW} < affinity < {AFFINITY_OUTLIER_THRESHOLD_HIGH} kcal/mol)")
    print(f"  Current: {n_usable} usable samples")

    if n_usable >= min_target:
        print(f"  Status: READY (sufficient data)")
        n_additional = 0
    else:
        n_needed = min_target - n_usable
        # 外れ値率を考慮（現在のデータから推定、または50%と仮定）
        if n_existing > 0:
            outlier_rate = n_outliers / n_existing
        else:
            outlier_rate = 0.5  # デフォルト推定
        # 必要数を外れ値率で補正
        estimated_docking = int(n_needed / max(1 - outlier_rate, 0.2))
        print(f"  Status: Need ~{estimated_docking} more docking runs (estimated outlier rate: {outlier_rate*100:.0f}%)")
        n_additional = estimated_docking

    print()

    # Step 3: Active Learning設定
    print("[Step 3] Active Learning settings")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Docking: N={args.runs}, n={args.evals}")
    print()

    # 推定時間
    n_al_docking = args.iterations * args.batch_size
    n_new_docking = n_additional + n_al_docking
    time_per_seq = 10 if args.quick else (60 if args.runs == 1 else 300)
    est_time = n_new_docking * time_per_seq

    print("[Summary]")
    print(f"  Existing results: {n_existing} total, {n_usable} usable")
    print(f"  Target usable samples: {min_target}")
    print(f"  Estimated initial docking: ~{n_additional} runs")
    print(f"  AL iterations: {args.iterations} x {args.batch_size} = {n_al_docking}")
    print(f"  Total estimated docking: ~{n_new_docking}")
    print(f"  Estimated time: {est_time // 60} min ({est_time / 3600:.1f} hours)")
    print()
    print("  Note: Pipeline will automatically collect more data if needed")
    print("        to reach the target number of usable samples.")
    print()

    # 最終確認
    if not args.no_confirm:
        response = input("Start? [y/N]: ")
        if response.lower() != 'y':
            return None

    return {
        "existing_results": existing_results,
        "n_additional": 0,  # run()メソッドで自動的に必要数を収集
        "n_existing": n_existing,
        "n_usable": n_usable
    }


def main():
    args = parse_args()

    # プリセット適用
    if args.quick:
        args.runs = 1
        args.evals = 5000
        args.timeout = 120
        args.iterations = 2
        args.batch_size = 2
        args.min_usable_samples = 20  # クイックテスト用に少なく
    elif args.production:
        args.runs = 5
        args.evals = 100000
        args.timeout = 1200
        args.iterations = 10
        args.batch_size = 5

    # 受容体ファイル確認
    if not Path(args.receptor).exists():
        print(f"ERROR: Receptor file not found: {args.receptor}")
        print("Create it with:")
        print("  cd Research_Linux/docking_setup")
        print("  micromamba run -n adcpsuite agfr -r ../Input/1O6K_noligand.pdb --toPdbqt")
        print("  micromamba run -n adcpsuite agfr -r 1O6K_noligand_rec.pdbqt -l ../Input/Peptideligand.pdbqt -asv 1.1 -o 1O6K")
        return

    # インタラクティブセットアップ
    setup = interactive_setup(args)
    if setup is None:
        print("Aborted.")
        return

    # パイプライン実行
    pipeline = ActiveLearningFromLead(
        lead_sequence=args.lead,
        receptor_file=args.receptor,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        al_result_dir=args.al_result_dir,
        existing_result_dir=args.existing_dir,
        n_init=setup["n_additional"],  # 追加サンプル数
        n_iterations=args.iterations,
        batch_size=args.batch_size,
        n_runs=args.runs,
        n_evals=args.evals,
        timeout=args.timeout,
        use_omm=not args.no_omm_rescoring,
        acquisition=args.acquisition,
        seed=args.seed,
        verbose=args.verbose,
        use_existing=True,
        min_existing=0,
        preloaded_results=setup["existing_results"],
        auto_select_model=args.auto_select_model or args.no_confirm,
        min_usable_samples=args.min_usable_samples
    )

    pipeline.run()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
