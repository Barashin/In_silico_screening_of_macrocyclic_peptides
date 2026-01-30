"""
AutoDock CrankPep Interface
============================

AutoDock CrankPepのドッキングシミュレーションを実行するためのインターフェース

Usage:
    # Google Colab/Linux環境での使用
    from adcp_interface import ADCPRunner

    runner = ADCPRunner(
        receptor_file="/path/to/receptor.trg",
        work_dir="/path/to/work_dir"
    )

    # 単一配列のドッキング
    result = runner.run_docking("ACDEFGHIKLMN")

    # バッチドッキング
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
    """ドッキング結果を格納するデータクラス"""
    sequence: str
    affinity: float  # Best affinity (kcal/mol)
    best_energy: float  # Best energy from MC search
    n_clusters: int  # Number of clusters
    status: str  # success, failed, error
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    pdb_file: Optional[str] = None
    dlg_file: Optional[str] = None


class ADCPRunner:
    """AutoDock CrankPepのランナークラス"""

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
        Parameters
        ----------
        receptor_file : str
            受容体の.trgファイルパス
        work_dir : str
            作業ディレクトリ
        adcp_path : str, optional
            ADCPの実行ファイルパス（Noneの場合は環境から自動検出）
        n_runs : int
            MCサーチの実行回数（デフォルト: 5）
        n_evals : int
            各MCサーチの評価回数（デフォルト: 100000）
        n_cores : int
            使用するCPUコア数（デフォルト: 2）
        omm_minimize : bool
            OpenMMによるエネルギー最小化を行うか（デフォルト: True）
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
        """ADCPコマンドを構築"""
        cmd_parts = [
            self.adcp_path,
            "-O",  # Overwrite existing files
            f"-T {self.receptor_file}",  # Target file (uppercase T)
            f'-s "{sequence}"',  # Sequence
            f"-o result_{job_name}",  # Output prefix
            f"-N {self.n_runs}",  # Number of runs
            f"-n {self.n_evals}",  # Number of evaluations
            "-cyc",  # Cyclic peptide (not -c)
            f"-w {output_dir}"  # Working directory (REQUIRED!)
        ]

        # Note: -j option may not be available in all ADCP versions
        # Using default parallelization instead

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
# Google Colab Helper Functions
# ============================================================================

def setup_adcp_colab():
    """Google ColabでADCPをセットアップするためのヘルパー関数"""
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
