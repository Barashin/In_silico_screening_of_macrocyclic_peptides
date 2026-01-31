"""
AutoDock CrankPep Interface
============================

【概要】
AutoDock CrankPep (ADCP) のドッキングシミュレーションを実行するためのインターフェース

【AutoDock CrankPepとは？】
環状ペプチド（輪っか状のペプチド）専用のドッキングツールです。
ペプチドがタンパク質（受容体）にどのように結合するかをシミュレーションし、
結合力（affinity、単位: kcal/mol）を予測します。

【ドッキングの基本】
1. 受容体（Target）: 薬が結合する標的タンパク質
2. リガンド（Ligand）: 結合する分子（ここではペプチド）
3. ドッキング: リガンドを受容体の結合部位に配置し、最適な結合ポーズを探索

【ADCPの特徴】
- 環状ペプチド専用に最適化（-cyc オプション）
- モンテカルロ法による柔軟なポーズ探索
- OpenMMによる精密なエネルギー計算（リスコアリング）

【出力ファイル】
- *_summary.dlg: ドッキング結果のサマリー（affinityはここから読み取る）
- *_out.pdb: 最良ポーズの3D構造

【このファイルの役割】
ADCPコマンドをPythonから実行し、結果を解析するためのラッパーです。
Active Learningパイプラインから呼び出されます。

Usage:
    # 基本的な使い方
    from adcp_interface import ADCPRunner

    runner = ADCPRunner(
        receptor_file="/path/to/receptor.trg",  # 受容体ファイル
        work_dir="/path/to/work_dir"  # 結果の保存先
    )

    # 単一配列のドッキング
    result = runner.run_docking("ACDEFGHIKLMN")
    print(f"Affinity: {result.affinity} kcal/mol")

    # バッチドッキング（複数配列を一括処理）
    results = runner.batch_docking(["ACDEFGHIKLMN", "MLKINGHFEDCA"])
"""

import os
import re
import subprocess
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class DockingResult:
    """
    ドッキング結果を格納するデータクラス

    【各フィールドの説明】
    - sequence: ドッキングしたペプチド配列
    - affinity: 結合力（kcal/mol）。負の値が大きいほど強い結合
               例: -10 kcal/mol は -5 kcal/mol より強い
    - best_energy: モンテカルロ探索で見つかった最良エネルギー
    - n_clusters: 似たポーズをグループ化したクラスター数
               クラスターが多い = 複数の結合モードがある
    - status: ドッキングの成否
               "success": 正常終了
               "failed": ADCPがエラー終了
               "timeout": 時間切れ
               "error": その他のエラー
    """
    sequence: str
    affinity: float  # Best affinity (kcal/mol) - 結合力。負の値が大きいほど強い
    best_energy: float  # Best energy from MC search - モンテカルロ探索の最良エネルギー
    n_clusters: int  # Number of clusters - クラスター数（似たポーズのグループ数）
    status: str  # success, failed, timeout, error
    error_message: Optional[str] = None  # エラー時のメッセージ
    output_dir: Optional[str] = None  # 出力ディレクトリ
    pdb_file: Optional[str] = None  # 最良ポーズの3D構造ファイル
    dlg_file: Optional[str] = None  # ドッキングログファイル


class ADCPRunner:
    """
    AutoDock CrankPepのランナークラス

    【このクラスの役割】
    ADCPコマンドを構築・実行し、結果を解析します。
    1. ペプチド配列を受け取る
    2. ADCPコマンドを構築
    3. サブプロセスで実行
    4. 結果ファイル（.dlg）からaffinityを抽出

    【ドッキングの流れ】
    1. 受容体（.trg）とペプチド配列を入力
    2. ADCPがモンテカルロ法で結合ポーズを探索
    3. 複数回（n_runs）探索して最良のポーズを選択
    4. OpenMMで精密化（オプション）
    5. 結果を.dlgファイルに出力

    【計算時間の目安】
    - Quick (n_runs=1, n_evals=5000): 約30秒〜1分
    - Standard (n_runs=5, n_evals=100000): 約5〜10分
    - Production (n_runs=10, n_evals=500000): 約30分〜1時間
    """

    def __init__(
        self,
        receptor_file: str,
        work_dir: str,
        adcp_path: str = None,
        n_runs: int = 5,
        n_evals: int = 100000,
        n_cores: int = 2,
        omm_minimize: bool = True,
        omm_max_itr: int = 5
    ):
        """
        ADCPランナーを初期化

        Parameters
        ----------
        receptor_file : str
            受容体の.trgファイルパス
            ※.trgファイルはagfrコマンドで事前に作成する必要があります
            例: agfr -r receptor.pdbqt -l ligand.pdbqt -asv 1.1 -o target
        work_dir : str
            作業ディレクトリ（結果ファイルの保存先）
        adcp_path : str, optional
            ADCPの実行ファイルパス
            Noneの場合、環境変数やよくあるパスから自動検出
        n_runs : int
            モンテカルロ探索の実行回数（デフォルト: 5）
            多いほど精度が上がるが時間もかかる
        n_evals : int
            各探索での評価回数（デフォルト: 100000）
            多いほど広く探索するが時間もかかる
        n_cores : int
            使用するCPUコア数（デフォルト: 2）
        omm_minimize : bool
            OpenMMによるエネルギー最小化を行うか（デフォルト: True）
            より精密な結合エネルギーを計算できるが時間がかかる
        omm_max_itr : int
            OpenMMの最大イテレーション数（デフォルト: 5）
        """
        self.receptor_file = receptor_file
        self.work_dir = work_dir
        self.adcp_path = adcp_path
        self.n_runs = n_runs
        self.n_evals = n_evals
        self.n_cores = n_cores
        self.omm_minimize = omm_minimize
        self.omm_max_itr = omm_max_itr

        # 作業ディレクトリの作成
        os.makedirs(work_dir, exist_ok=True)

        # ADCPパスの検出
        if self.adcp_path is None:
            self.adcp_path = self._detect_adcp_path()

    def _detect_adcp_path(self) -> str:
        """ADCPの実行パスを検出"""
        # whichコマンドで検索（最優先）
        try:
            result = subprocess.run(['which', 'adcp'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # 一般的なインストールパス
        possible_paths = [
            # Linux micromamba環境
            os.path.expanduser("~/micromamba/envs/adcpsuite/bin/adcp"),
            # miniforge環境
            os.path.expanduser("~/miniforge3/envs/adcpsuite/bin/adcp"),
            # Conda環境
            os.path.expanduser("~/miniconda3/envs/adcpsuite/bin/adcp"),
            os.path.expanduser("~/anaconda3/envs/adcpsuite/bin/adcp"),
            # Google Colabでのmicromamba環境
            "/root/micromamba/envs/adcpsuite/bin/adcp",
            # システムパス
            "/usr/local/bin/adcp"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "ADCPが見つかりません。adcp_pathを明示的に指定するか、"
            "micromamba/condaでadcpsuiteをインストールしてください"
        )

    def run_docking(self, sequence: str, job_name: str = None) -> DockingResult:
        """
        単一配列のドッキングを実行

        Parameters
        ----------
        sequence : str
            アミノ酸配列（1文字表記、例: "ACDEFGHIKLMN"）
        job_name : str, optional
            ジョブ名（Noneの場合は配列から自動生成）

        Returns
        -------
        DockingResult
            ドッキング結果
        """
        if job_name is None:
            job_name = sequence.lower()

        output_dir = os.path.join(self.work_dir, job_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # ADCPコマンドの構築
            cmd = self._build_adcp_command(sequence, job_name, output_dir)

            # 実行（cwdは不要、-w オプションで指定）
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1時間タイムアウト
            )

            if result.returncode != 0:
                return DockingResult(
                    sequence=sequence,
                    affinity=float('inf'),
                    best_energy=float('inf'),
                    n_clusters=0,
                    status='failed',
                    error_message=result.stderr,
                    output_dir=output_dir
                )

            # 結果のパース
            dlg_file = os.path.join(output_dir, f"result_{job_name}_summary.dlg")
            pdb_file = os.path.join(output_dir, f"result_{job_name}_out.pdb")

            affinity, best_energy, n_clusters = self._parse_dlg_file(dlg_file)

            return DockingResult(
                sequence=sequence,
                affinity=affinity,
                best_energy=best_energy,
                n_clusters=n_clusters,
                status='success',
                output_dir=output_dir,
                pdb_file=pdb_file if os.path.exists(pdb_file) else None,
                dlg_file=dlg_file if os.path.exists(dlg_file) else None
            )

        except subprocess.TimeoutExpired:
            return DockingResult(
                sequence=sequence,
                affinity=float('inf'),
                best_energy=float('inf'),
                n_clusters=0,
                status='timeout',
                error_message='Docking timeout after 1 hour',
                output_dir=output_dir
            )

        except Exception as e:
            return DockingResult(
                sequence=sequence,
                affinity=float('inf'),
                best_energy=float('inf'),
                n_clusters=0,
                status='error',
                error_message=str(e),
                output_dir=output_dir
            )

    def _build_adcp_command(self, sequence: str, job_name: str, output_dir: str) -> str:
        """
        ADCPコマンドを構築

        【ADCPの主要オプション】
        -O: 既存ファイルを上書き
        -T: ターゲット（受容体）ファイル
        -s: ペプチド配列（アミノ酸1文字表記）
        -o: 出力ファイルのプレフィックス
        -N: モンテカルロ探索の実行回数
        -n: 各探索での評価ステップ数
        -cyc: 環状ペプチドとして処理（重要！）
        -w: 作業ディレクトリ
        -e: OpenMMによるエネルギー最小化設定
        """
        cmd_parts = [
            self.adcp_path,
            "-O",  # Overwrite: 既存ファイルを上書き
            f"-T {self.receptor_file}",  # Target: 受容体ファイル（.trg）
            f'-s "{sequence}"',  # Sequence: ペプチド配列
            f"-o result_{job_name}",  # Output: 出力ファイル名のプレフィックス
            f"-N {self.n_runs}",  # Number of runs: MC探索の実行回数
            f"-n {self.n_evals}",  # Number of evaluations: 各探索の評価回数
            "-cyc",  # Cyclic: 環状ペプチドモード（N末とC末を結合）
            f"-w {output_dir}"  # Working directory: 結果の保存先
        ]

        # OpenMMによるエネルギー最小化（より精密な計算）
        # in-vacuo: 真空中での計算、数字は最大イテレーション数
        if self.omm_minimize:
            cmd_parts.append(f"-e in-vacuo,{self.omm_max_itr}")

        return " ".join(cmd_parts)

    def _parse_dlg_file(self, dlg_file: str) -> Tuple[float, float, int]:
        """DLGファイルから結果をパース"""
        affinity = float('inf')
        best_energy = float('inf')
        n_clusters = 0

        if not os.path.exists(dlg_file):
            return affinity, best_energy, n_clusters

        with open(dlg_file, 'r') as f:
            content = f.read()

            # Best affinityの抽出（mode 1の値）
            affinity_match = re.search(r'^\s+1\s+([-\d\.]+)', content, re.MULTILINE)
            if affinity_match:
                affinity = float(affinity_match.group(1))

            # Best energyの抽出
            energy_match = re.search(r'bestEnergy in run \d+ ([-\d\.]+)', content)
            if energy_match:
                best_energy = float(energy_match.group(1))

            # クラスター数のカウント
            n_clusters = len(re.findall(r'^\s+\d+\s+[-\d\.]+', content, re.MULTILINE))

        return affinity, best_energy, n_clusters

    def batch_docking(
        self,
        sequences: List[str],
        parallel: bool = False,
        max_workers: int = 4
    ) -> List[DockingResult]:
        """
        複数配列のバッチドッキング

        Parameters
        ----------
        sequences : List[str]
            アミノ酸配列のリスト
        parallel : bool
            並列実行するか（デフォルト: False）
        max_workers : int
            並列実行時の最大ワーカー数（デフォルト: 4）

        Returns
        -------
        List[DockingResult]
            ドッキング結果のリスト
        """
        results = []

        if parallel:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_seq = {
                    executor.submit(self.run_docking, seq): seq
                    for seq in sequences
                }

                for future in as_completed(future_to_seq):
                    result = future.result()
                    results.append(result)
                    print(f"Completed: {result.sequence}, Affinity: {result.affinity:.2f}")
        else:
            for i, seq in enumerate(sequences):
                print(f"Running docking {i+1}/{len(sequences)}: {seq}")
                result = self.run_docking(seq)
                results.append(result)
                print(f"  Affinity: {result.affinity:.2f} kcal/mol")

        return results

    def results_to_dataframe(self, results: List[DockingResult]):
        """結果をDataFrameに変換（pandasが必要）"""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for results_to_dataframe()")

        data = []
        for r in results:
            data.append({
                'sequence': r.sequence,
                'affinity': r.affinity,
                'best_energy': r.best_energy,
                'n_clusters': r.n_clusters,
                'status': r.status,
                'error_message': r.error_message,
                'output_dir': r.output_dir,
                'pdb_file': r.pdb_file,
                'dlg_file': r.dlg_file
            })
        return pd.DataFrame(data)


class ADCPActiveLearningIntegration:
    """
    Active LearningパイプラインとADCPを統合するクラス

    Usage:
        from adcp_interface import ADCPActiveLearningIntegration
        from active_learning_pipeline import ActiveLearningPipeline, ActiveLearningConfig

        # 設定
        config = ActiveLearningConfig(...)
        pipeline = ActiveLearningPipeline(config)

        # ADCP統合
        integration = ADCPActiveLearningIntegration(
            pipeline=pipeline,
            receptor_file="/path/to/receptor.trg"
        )

        # 実行（実際のドッキングを使用）
        integration.run_with_real_docking(n_iterations=10)
    """

    def __init__(
        self,
        pipeline,  # ActiveLearningPipeline
        receptor_file: str,
        adcp_path: str = None,
        n_runs: int = 5,
        n_evals: int = 100000
    ):
        """
        Parameters
        ----------
        pipeline : ActiveLearningPipeline
            Active Learningパイプライン
        receptor_file : str
            受容体の.trgファイル
        adcp_path : str, optional
            ADCPのパス
        n_runs : int
            MCサーチ回数
        n_evals : int
            評価回数
        """
        self.pipeline = pipeline

        # ADCPランナーの初期化
        self.adcp_runner = ADCPRunner(
            receptor_file=receptor_file,
            work_dir=os.path.join(pipeline.config.output_dir, "docking_results"),
            adcp_path=adcp_path,
            n_runs=n_runs,
            n_evals=n_evals
        )

    def run_real_docking(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """
        実際のADCPドッキングを実行

        Parameters
        ----------
        sequences : List[str]
            配列のリスト

        Returns
        -------
        List[Tuple[str, float]]
            (配列, affinity)のリスト
        """
        results = self.adcp_runner.batch_docking(sequences)

        observations = []
        for r in results:
            if r.status == 'success':
                observations.append((r.sequence, r.affinity))
            else:
                print(f"Warning: Docking failed for {r.sequence}: {r.error_message}")

        return observations

    def run_with_real_docking(self, n_iterations: int = None):
        """
        実際のドッキングを使用してActive Learningを実行

        Parameters
        ----------
        n_iterations : int, optional
            イテレーション数
        """
        if n_iterations is None:
            n_iterations = self.pipeline.config.n_iterations

        # パイプラインのドッキング関数を置き換え
        original_simulate_docking = self.pipeline.simulate_docking
        self.pipeline.simulate_docking = self.run_real_docking

        try:
            # パイプライン実行
            history = self.pipeline.run(n_iterations)
            return history
        finally:
            # 元に戻す
            self.pipeline.simulate_docking = original_simulate_docking


# ============================================================================
# Google Colab Helper Functions（Google Colab用ヘルパー関数）
# ============================================================================
# Google Colabでは、ADCPがデフォルトでインストールされていないため、
# micromambaを使ってインストールする必要があります。
#
# 【Colabでの使用手順】
# 1. setup_adcp_colab() のスクリプトを実行してADCPをインストール
# 2. prepare_receptor_colab() で受容体ファイルを準備
# 3. run_adcp_colab() または batch_docking_colab() でドッキング実行

def setup_adcp_colab():
    """
    Google ColabでADCPをセットアップするためのヘルパー関数

    【この関数の役割】
    Colabにmicromambaをインストールし、adcpsuiteパッケージをセットアップする
    シェルスクリプトを生成します。

    【使い方】
    script = setup_adcp_colab()
    !bash -c "$script"  # Colabで実行
    """
    setup_script = """
#!/bin/bash
# Install micromamba
curl -Ls https://micro.mamba.pm/install.sh | bash

# Set PATH
export PATH=/root/.local/bin:$PATH

# Download and run adcpsuite setup script
wget -q https://ccsb.scripps.edu/mamba/scripts/adcpsuite_micromamba.sh

# Fix for Colab compatibility
sed -i 's|\\. ${MAMBA_ROOT_PREFIX}/etc/profile.d/micromamba.sh|eval "$(${MAMBA_EXE} shell hook --shell=bash --root-prefix=${MAMBA_ROOT_PREFIX})"|' adcpsuite_micromamba.sh

# Install adcpsuite
export MAMBA_EXE=/root/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/root/micromamba
bash adcpsuite_micromamba.sh

# Clean up
rm -rf /root/AdcpTmpDir

echo "ADCP setup complete!"
"""
    print("Run the following commands to setup ADCP on Google Colab:")
    print("="*60)
    print(setup_script)
    print("="*60)

    return setup_script


def prepare_receptor_colab(receptor_pdb: str, ligand_pdbqt: str, output_prefix: str = "target") -> str:
    """
    Google Colabで受容体の準備とターゲットファイル作成のコマンドを生成

    Parameters
    ----------
    receptor_pdb : str
        受容体PDBファイルのパス
    ligand_pdbqt : str
        リガンドPDBQTファイルのパス（結合部位特定用）
    output_prefix : str
        出力ファイルのプレフィックス

    Returns
    -------
    str
        実行コマンド
    """
    cmd = f"""
# Step 1: Convert receptor PDB to PDBQT
micromamba run -n adcpsuite agfr -r {receptor_pdb} --toPdbqt

# Step 2: Create target file (binding site detection)
# -asv 1.1: AutoSite volume setting
receptor_pdbqt="{receptor_pdb.replace('.pdb', '_rec.pdbqt')}"
micromamba run -n adcpsuite agfr -r $receptor_pdbqt -l {ligand_pdbqt} -asv 1.1 -o {output_prefix}

echo "Generated files:"
ls -la {output_prefix}*
"""
    return cmd


def run_adcp_colab(
    target_file: str,
    sequence: str,
    output_dir: str,
    n_runs: int = 5,
    n_evals: int = 100000
) -> str:
    """
    Google Colabでの単純なADCP実行コマンドを生成

    Parameters
    ----------
    target_file : str
        ターゲットファイル(.trg)のパス
    sequence : str
        ペプチド配列
    output_dir : str
        出力ディレクトリ
    n_runs : int
        MCサーチの実行回数
    n_evals : int
        評価ステップ数

    Returns
    -------
    str
        実行コマンド
    """
    seq_lower = sequence.lower()
    cmd = f"""
# Create output directory
mkdir -p {output_dir}/{seq_lower}

# Run ADCP docking
micromamba run -n adcpsuite adcp -O \\
    -T {target_file} \\
    -s "{sequence}" \\
    -cyc \\
    -N {n_runs} \\
    -n {n_evals} \\
    -o result_{seq_lower} \\
    -w {output_dir}/{seq_lower}

# Show results
echo "Results:"
cat {output_dir}/{seq_lower}/result_{seq_lower}_summary.dlg | head -40
"""
    return cmd


def batch_docking_colab(
    target_file: str,
    sequences: List[str],
    output_dir: str,
    n_runs: int = 5,
    n_evals: int = 100000
) -> str:
    """
    Google Colabでのバッチドッキングコマンドを生成

    Parameters
    ----------
    target_file : str
        ターゲットファイル(.trg)のパス
    sequences : List[str]
        ペプチド配列のリスト
    output_dir : str
        出力ディレクトリ
    n_runs : int
        MCサーチの実行回数
    n_evals : int
        評価ステップ数

    Returns
    -------
    str
        実行コマンド
    """
    commands = [f"# Batch docking for {len(sequences)} sequences\n"]

    for i, seq in enumerate(sequences):
        seq_lower = seq.lower()
        cmd = f"""
echo "[{i+1}/{len(sequences)}] Docking: {seq}"
mkdir -p {output_dir}/{seq_lower}
micromamba run -n adcpsuite adcp -O -T {target_file} -s "{seq}" -cyc -N {n_runs} -n {n_evals} -o result_{seq_lower} -w {output_dir}/{seq_lower}
"""
        commands.append(cmd)

    return "\n".join(commands)


# ============================================================================
# Test Functions
# ============================================================================

def test_parse_dlg():
    """DLGパーサーのテスト"""
    dlg_content = """
performing MC searches with: /path/to/CrankiteAD
target data from file: /content/1O6K.trg
job name: result_test, summary file result_test_summary.dlg
bestEnergies [-8.8724, -10.1737, -2.66824, -11.6523, 82.7855]
bestEnergy in run 4 -11.652300 (0)
mode |  affinity  | ref. | clust. | rmsd | energy | best |
     | (kcal/mol) | fnc  |  size  | stdv |  stdv  | run  |
-----+------------+------+--------+------+--------+------+
   1         -6.9      0.0       5      NA      NA    164
   2         -6.6      0.0      13      NA      NA    150
   3         -6.0      0.0       5      NA      NA    086
"""
    # テスト用のパース
    affinity_match = re.search(r'^\s+1\s+([-\d\.]+)', dlg_content, re.MULTILINE)
    if affinity_match:
        print(f"Parsed affinity: {affinity_match.group(1)}")

    energy_match = re.search(r'bestEnergy in run \d+ ([-\d\.]+)', dlg_content)
    if energy_match:
        print(f"Parsed best energy: {energy_match.group(1)}")


if __name__ == "__main__":
    test_parse_dlg()
