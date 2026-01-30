#!/usr/bin/env python
"""
シンプルなActive Learning + 実ドッキングテスト

Usage:
    # テスト実行（デフォルト設定）
    python simple_al_docking_test.py

    # イテレーションとバッチサイズを指定
    python simple_al_docking_test.py --iterations 3 --batch-size 5

    # 本番設定（高精度ドッキング）
    python simple_al_docking_test.py --production

    # カスタム設定
    python simple_al_docking_test.py -i 5 -b 3 -N 3 -n 50000 --acquisition UCB
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

# パス追加
sys.path.insert(0, '/home/shizuku/In_silico_RaPID/Research_Linux/Research')

from active_learning_gnn import Config, ActiveLearningPipeline, CustomGNN, SurrogateModel
from transformer_models import SequenceTransformerEncoder, GraphTransformerEncoder, CNN1DEncoder, CatBoostSurrogate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# モデルタイプのリスト（GNN + Transformer + CNN + CatBoost）
GNN_TYPES = ["GIN", "GCN", "GAT", "GraphSAGE", "MPNN", "SeqTransformer", "GraphTransformer", "CNN1D", "CatBoost"]


def create_parity_plot(y_true_train, y_pred_train, y_true_test, y_pred_test,
                       y_std_train=None, y_std_test=None,
                       output_path=None, iteration=None):
    """
    パリティプロット（予測値 vs 実測値）を作成して保存
    TrainデータとTestデータを異なる色で表示

    Parameters
    ----------
    y_true_train : array-like
        Train実測値
    y_pred_train : array-like
        Train予測値
    y_true_test : array-like
        Test実測値
    y_pred_test : array-like
        Test予測値
    y_std_train : array-like, optional
        Train予測の標準偏差（エラーバー用）
    y_std_test : array-like, optional
        Test予測の標準偏差（エラーバー用）
    output_path : str, optional
        保存先パス
    iteration : int, optional
        イテレーション番号（タイトル用）
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
    if y_std_train is not None:
        y_std_train = np.array(y_std_train)
        ax.errorbar(y_true_train, y_pred_train, yerr=y_std_train, fmt='o',
                   alpha=0.5, capsize=2, markersize=5, color='blue',
                   label=f'Train (n={len(y_true_train)})')
    else:
        ax.scatter(y_true_train, y_pred_train, alpha=0.5, s=40, color='blue',
                  label=f'Train (n={len(y_true_train)})')

    # Testデータポイント（赤）
    if y_std_test is not None:
        y_std_test = np.array(y_std_test)
        ax.errorbar(y_true_test, y_pred_test, yerr=y_std_test, fmt='s',
                   alpha=0.8, capsize=3, markersize=7, color='red',
                   label=f'Test (n={len(y_true_test)})')
    else:
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

    # 軸設定
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_xlabel('Actual Affinity (kcal/mol)', fontsize=12)
    ax.set_ylabel('Predicted Affinity (kcal/mol)', fontsize=12)

    # タイトル
    if iteration is not None:
        title = f'Parity Plot - Iteration {iteration}'
    else:
        title = 'Parity Plot - GNN Surrogate Model'
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

    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'    Parity plot saved: {output_path}')

    plt.close()

    return {
        'r2_train': r2_train, 'rmse_train': rmse_train, 'mae_train': mae_train, 'n_train': len(y_true_train),
        'r2_test': r2_test, 'rmse_test': rmse_test, 'mae_test': mae_test, 'n_test': len(y_true_test)
    }


def compare_gnn_models(sequences, affinities, config, output_dir):
    """
    複数のGNN/Transformerアーキテクチャを比較し、パリティプロットを作成

    Parameters
    ----------
    sequences : list
        配列のリスト
    affinities : np.ndarray
        親和性の配列
    config : Config
        設定
    output_dir : str
        出力ディレクトリ

    Returns
    -------
    dict
        各モデルタイプの評価結果
    """
    print(f'\n{"="*70}')
    print('Comparing GNN/Transformer Architectures')
    print(f'{"="*70}')

    # Train/Test分割
    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences, affinities,
        test_size=0.2,
        random_state=config.seed
    )
    print(f'\nData split: Train={len(seq_train)}, Test={len(seq_test)}')

    results = {}
    # 9モデル + 1サマリー = 10 → 2x5レイアウト
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for idx, model_type in enumerate(GNN_TYPES):
        print(f'\n[{idx+1}/{len(GNN_TYPES)}] Training {model_type}...')

        try:
            # モデル作成（GNN, Transformer, CNN, or CatBoost）
            if model_type == "SeqTransformer":
                # Sequence Transformer
                encoder = SequenceTransformerEncoder(
                    d_model=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    num_layers=config.gnn_num_layers,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="sequence")
            elif model_type == "GraphTransformer":
                # Graph Transformer
                encoder = GraphTransformerEncoder(
                    in_channels=6,
                    hidden_channels=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    num_layers=config.gnn_num_layers,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="graph")
            elif model_type == "CNN1D":
                # 1D-CNN
                encoder = CNN1DEncoder(
                    embed_dim=config.gnn_hidden_dim // 2,
                    num_filters=config.gnn_hidden_dim,
                    out_channels=config.gnn_output_dim,
                    dropout=config.gnn_dropout
                )
                surrogate = SurrogateModel(encoder, config, encoder_mode="sequence")
            elif model_type == "CatBoost":
                # CatBoost（独自のサロゲートモデル）
                surrogate = CatBoostSurrogate(
                    n_estimators=500,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=config.seed,
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
                # 評価指標
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

                # Trainデータ
                ax.scatter(y_train[valid_train], y_pred_train[valid_train],
                          alpha=0.5, s=30, color='blue', label=f'Train (n={valid_train.sum()})')
                # Testデータ
                ax.scatter(y_test[valid_test], y_pred_test[valid_test],
                          alpha=0.8, s=50, color='red', marker='s', label=f'Test (n={valid_test.sum()})')

                # 理想線
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
            axes[idx].text(0.5, 0.5, f'{model_type}\nError: {str(e)[:30]}',
                          ha='center', va='center', transform=axes[idx].transAxes)

    # 最後のサブプロット（10番目）は比較サマリー
    ax = axes[9]
    ax.axis('off')

    # 結果サマリーテーブル
    summary_text = "Model Comparison Summary\n" + "="*35 + "\n\n"
    summary_text += f"{'Type':<15} {'Train R²':>10} {'Test R²':>10}\n"
    summary_text += "-"*37 + "\n"

    valid_results = {k: v for k, v in results.items() if 'r2_test' in v}
    for model_type in GNN_TYPES:
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

    plt.suptitle('GNN/Transformer Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    comparison_path = os.path.join(output_dir, 'parity_plots', 'model_comparison.png')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f'\nComparison plot saved: {comparison_path}')
    plt.close()

    return results


def select_gnn_model(results):
    """
    ユーザーにモデルを選択させる（GNN/Transformer対応）

    Parameters
    ----------
    results : dict
        各モデルタイプの評価結果

    Returns
    -------
    str
        選択されたモデルタイプ
    """
    valid_results = {k: v for k, v in results.items() if 'r2_test' in v}

    if not valid_results:
        print("No valid models available. Using default GIN.")
        return "GIN"

    # Test R²でソート
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['r2_test'], reverse=True)

    print(f'\n{"="*70}')
    print('Select Model Architecture')
    print(f'{"="*70}')
    print(f'\n{"No.":<4} {"Type":<16} {"Train R²":>10} {"Test R²":>10} {"Test RMSE":>12}')
    print("-"*55)

    for i, (model_type, r) in enumerate(sorted_results, 1):
        marker = " <-- Best" if i == 1 else ""
        print(f'{i:<4} {model_type:<16} {r["r2_train"]:>10.4f} {r["r2_test"]:>10.4f} {r["rmse_test"]:>12.2f}{marker}')

    print("-"*50)
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


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='Active Learning with Real ADCP Docking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # テスト実行（デフォルト: 1 iteration, 2 sequences）
  %(prog)s

  # 3イテレーション、バッチサイズ5
  %(prog)s --iterations 3 --batch-size 5

  # 本番設定（高精度）
  %(prog)s --production

  # カスタム設定
  %(prog)s -i 5 -b 3 -N 3 -n 50000 --acquisition UCB

  # 受容体ファイルを指定
  %(prog)s --receptor /path/to/receptor.trg
        """
    )

    # Active Learning設定
    al_group = parser.add_argument_group('Active Learning Settings')
    al_group.add_argument(
        '-i', '--iterations',
        type=int,
        default=1,
        help='Number of Active Learning iterations (default: 1)'
    )
    al_group.add_argument(
        '-b', '--batch-size',
        type=int,
        default=2,
        help='Number of sequences to evaluate per iteration (default: 2)'
    )
    al_group.add_argument(
        '--acquisition',
        choices=['EI', 'UCB', 'PI'],
        default='EI',
        help='Acquisition function: EI (Expected Improvement), UCB (Upper Confidence Bound), PI (Probability of Improvement) (default: EI)'
    )
    al_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # ドッキング設定
    dock_group = parser.add_argument_group('Docking Settings')
    dock_group.add_argument(
        '-N', '--runs',
        type=int,
        default=1,
        help='Number of MC search runs (default: 1 for test, 5 for production)'
    )
    dock_group.add_argument(
        '-n', '--evals',
        type=int,
        default=5000,
        help='Number of evaluation steps per run (default: 5000 for test, 100000 for production)'
    )
    dock_group.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per docking in seconds (default: 60 for test, 3600 for production)'
    )
    dock_group.add_argument(
        '--omm-rescoring',
        action='store_true',
        default=True,
        help='Enable OpenMM rescoring for higher accuracy (default: True, recommended for production)'
    )
    dock_group.add_argument(
        '--no-omm-rescoring',
        action='store_false',
        dest='omm_rescoring',
        help='Disable OpenMM rescoring for faster execution (useful for quick tests)'
    )

    # ファイルパス設定
    path_group = parser.add_argument_group('File Paths')
    path_group.add_argument(
        '--receptor',
        type=str,
        default='/home/shizuku/In_silico_RaPID/Research_Linux/Research/docking_setup/1O6K.trg',
        help='Path to receptor target file (.trg)'
    )
    path_group.add_argument(
        '--result-dir',
        type=str,
        default='/home/shizuku/In_silico_RaPID/Research_Linux/Research/result',
        help='Path to result directory'
    )
    path_group.add_argument(
        '--output-dir',
        type=str,
        default='/home/shizuku/In_silico_RaPID/Research_Linux/Research/al_output',
        help='Path to output directory'
    )
    path_group.add_argument(
        '--work-dir',
        type=str,
        default='/home/shizuku/In_silico_RaPID/Research_Linux/Research/result/new_docking_test',
        help='Path to docking work directory'
    )

    # プリセット設定
    preset_group = parser.add_argument_group('Preset Configurations')
    preset_group.add_argument(
        '--production',
        action='store_true',
        help='Use production settings (N=5, n=100000, timeout=3600, iterations=10, batch-size=5)'
    )
    preset_group.add_argument(
        '--quick',
        action='store_true',
        help='Use quick test settings (N=1, n=1000, timeout=30, iterations=1, batch-size=1)'
    )

    # その他
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # プリセット適用
    if args.production:
        args.runs = 5
        args.evals = 100000
        args.timeout = 3600
        args.iterations = 10
        args.batch_size = 5
        print("Using PRODUCTION settings")
    elif args.quick:
        args.runs = 1
        args.evals = 1000
        args.timeout = 30
        args.iterations = 1
        args.batch_size = 1
        args.omm_rescoring = False  # Disable OpenMM for quick tests
        print("Using QUICK test settings")

    return args


def main():
    args = parse_args()

    print('='*70)
    print('Simple Active Learning + Real Docking Test')
    print('='*70)

    # ターゲットファイルの確認
    receptor_file = args.receptor

    if not os.path.exists(receptor_file):
        print(f"\nError: Receptor file not found: {receptor_file}")
        print("\nPlease create the target file first:")
        print("  cd Research_Linux/Research/docking_setup")
        print("  micromamba activate adcpsuite")
        print("  agfr -r ../Input/1O6K_noligand.pdb --toPdbqt")
        print("  agfr -r 1O6K_noligand_rec.pdbqt -l ../Input/Peptideligand.pdbqt -asv 1.1 -o 1O6K")
        sys.exit(1)

    # 設定
    config = Config(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        n_iterations=args.iterations,
        batch_size=args.batch_size,
        acquisition=args.acquisition,
        seed=args.seed
    )

    # 設定表示
    print(f'\n{"="*70}')
    print('Configuration')
    print(f'{"="*70}')
    print(f'\n[Active Learning]')
    print(f'  Iterations: {config.n_iterations}')
    print(f'  Batch size: {config.batch_size}')
    print(f'  Acquisition function: {config.acquisition}')
    print(f'  Random seed: {config.seed}')

    print(f'\n[Docking]')
    print(f'  MC runs (N): {args.runs}')
    print(f'  Evaluations per run (n): {args.evals}')
    print(f'  Timeout: {args.timeout} seconds')
    print(f'  OpenMM rescoring: {args.omm_rescoring} {"(high accuracy)" if args.omm_rescoring else "(faster, lower accuracy)"}')

    print(f'\n[Files]')
    print(f'  Receptor: {receptor_file}')
    print(f'  Result dir: {args.result_dir}')
    print(f'  Output dir: {args.output_dir}')
    print(f'  Work dir: {args.work_dir}')

    # 推定時間
    total_sequences = config.n_iterations * config.batch_size
    est_time_per_seq = args.evals / 20000  # 大まかな推定: 20000 evals = 1秒
    est_total_time = total_sequences * est_time_per_seq
    print(f'\n[Estimation]')
    print(f'  Total sequences to dock: {total_sequences}')
    print(f'  Estimated time per sequence: ~{est_time_per_seq:.1f} seconds')
    print(f'  Estimated total time: ~{est_total_time:.1f} seconds (~{est_total_time/60:.1f} minutes)')

    # 確認
    if not args.no_confirm:
        print(f'\n{"="*70}')
        try:
            response = input('Proceed with docking? (y/n): ')
            if response.lower() != 'y':
                print('Cancelled')
                sys.exit(0)
        except KeyboardInterrupt:
            print('\nCancelled')
            sys.exit(0)

    print(f'\n{"="*70}')
    print('Starting Active Learning Pipeline')
    print(f'{"="*70}')

    # パイプライン初期化
    print('\nInitializing pipeline...')
    pipeline = ActiveLearningPipeline(config)
    pipeline.initialize()

    # データ読み込み
    print('Loading initial data...')
    pipeline.load_data()
    print(f'  Loaded {len(pipeline.state.sequences)} sequences')
    print(f'  Initial best: {pipeline.state.best_seq} ({pipeline.state.best_aff:.2f} kcal/mol)')

    # GNNアーキテクチャ比較
    if len(pipeline.state.sequences) >= 10:
        gnn_results = compare_gnn_models(
            pipeline.state.sequences,
            np.array(pipeline.state.affinities),
            config,
            config.output_dir
        )

        # ユーザーにGNNを選択させる
        if args.no_confirm:
            # 自動選択（最良のTest R²）
            valid_results = {k: v for k, v in gnn_results.items() if 'r2_test' in v}
            if valid_results:
                selected_gnn = max(valid_results.keys(), key=lambda x: valid_results[x]['r2_test'])
                print(f'\nAuto-selected GNN: {selected_gnn} (best Test R²)')
            else:
                selected_gnn = "GIN"
                print(f'\nUsing default GNN: {selected_gnn}')
        else:
            selected_gnn = select_gnn_model(gnn_results)

        # 選択されたモデルでパイプラインのサロゲートモデルを更新
        print(f'\n{"="*70}')
        print(f'Reinitializing pipeline with {selected_gnn}...')
        print(f'{"="*70}')

        # 新しいモデルを作成（GNN or Transformer）
        if selected_gnn == "SeqTransformer":
            new_encoder = SequenceTransformerEncoder(
                d_model=config.gnn_hidden_dim,
                out_channels=config.gnn_output_dim,
                num_layers=config.gnn_num_layers,
                dropout=config.gnn_dropout
            )
            pipeline.surrogate = SurrogateModel(new_encoder, config, encoder_mode="sequence")
        elif selected_gnn == "GraphTransformer":
            new_encoder = GraphTransformerEncoder(
                in_channels=6,
                hidden_channels=config.gnn_hidden_dim,
                out_channels=config.gnn_output_dim,
                num_layers=config.gnn_num_layers,
                dropout=config.gnn_dropout
            )
            pipeline.surrogate = SurrogateModel(new_encoder, config, encoder_mode="graph")
        elif selected_gnn == "CNN1D":
            new_encoder = CNN1DEncoder(
                embed_dim=config.gnn_hidden_dim // 2,
                num_filters=config.gnn_hidden_dim,
                out_channels=config.gnn_output_dim,
                dropout=config.gnn_dropout
            )
            pipeline.surrogate = SurrogateModel(new_encoder, config, encoder_mode="sequence")
        elif selected_gnn == "CatBoost":
            pipeline.surrogate = CatBoostSurrogate(
                n_estimators=500,
                learning_rate=0.1,
                depth=6,
                random_seed=config.seed,
                verbose=False
            )
        else:
            new_encoder = CustomGNN(
                in_channels=6,
                hidden_channels=config.gnn_hidden_dim,
                out_channels=config.gnn_output_dim,
                num_layers=config.gnn_num_layers,
                dropout=config.gnn_dropout,
                conv_type=selected_gnn
            )
            pipeline.surrogate = SurrogateModel(new_encoder, config, encoder_mode="graph")
        print(f'  Surrogate model updated to use {selected_gnn}')
    else:
        print(f'\nNot enough data for GNN comparison (n={len(pipeline.state.sequences)}). Using default GIN.')
        selected_gnn = "GIN"

    # イテレーション履歴
    iteration_history = []

    # Active Learningループ
    for iteration in range(1, config.n_iterations + 1):
        print(f'\n{"="*70}')
        print(f'ITERATION {iteration}/{config.n_iterations}')
        print(f'{"="*70}')

        # 候補生成
        print(f'\n[{iteration}.1] Generating candidates...')
        pipeline.generate_candidates(n=100)
        print(f'    Generated {len(pipeline.state.candidates)} candidates')

        # サロゲートモデル学習
        print(f'[{iteration}.2] Training surrogate model...')
        pipeline.surrogate.fit(
            pipeline.state.sequences,
            np.array(pipeline.state.affinities),
            verbose=False
        )
        print(f'    ✓ Model trained on {len(pipeline.state.sequences)} sequences')

        # パリティプロット作成（Train/Test split 80/20）
        print(f'[{iteration}.2a] Creating parity plot (80/20 split)...')
        try:
            sequences = pipeline.state.sequences
            affinities = np.array(pipeline.state.affinities)

            # データが少なすぎる場合はスキップ
            if len(sequences) < 10:
                print(f'    ✗ Not enough data for train/test split (n={len(sequences)})')
            else:
                # 80/20でTrain/Test分割
                seq_train, seq_test, y_train, y_test = train_test_split(
                    sequences, affinities,
                    test_size=0.2,
                    random_state=config.seed
                )

                # Trainデータでモデルを再学習
                pipeline.surrogate.fit(seq_train, y_train, verbose=False)

                # Train/Testデータで予測
                y_pred_train, y_std_train = pipeline.surrogate.predict(seq_train)
                y_pred_test, y_std_test = pipeline.surrogate.predict(seq_test)

                # 有効な予測のみ使用
                valid_train = ~np.isnan(y_pred_train) & ~np.isnan(y_train)
                valid_test = ~np.isnan(y_pred_test) & ~np.isnan(y_test)

                if valid_train.sum() > 0 and valid_test.sum() > 0:
                    plot_path = os.path.join(
                        config.output_dir,
                        'parity_plots',
                        f'parity_iter_{iteration:03d}.png'
                    )
                    metrics = create_parity_plot(
                        y_train[valid_train], y_pred_train[valid_train],
                        y_test[valid_test], y_pred_test[valid_test],
                        y_std_train[valid_train] if y_std_train is not None else None,
                        y_std_test[valid_test] if y_std_test is not None else None,
                        output_path=plot_path,
                        iteration=iteration
                    )
                    print(f'    Train: {metrics["n_train"]}, Test: {metrics["n_test"]}')
                    print(f'    Train R²={metrics["r2_train"]:.4f}, Test R²={metrics["r2_test"]:.4f}')
                    print(f'    Test RMSE={metrics["rmse_test"]:.2f}, MAE={metrics["mae_test"]:.2f}')

                    # 全データで再学習（次のステップのため）
                    pipeline.surrogate.fit(sequences, affinities, verbose=False)
                else:
                    print(f'    ✗ No valid predictions for parity plot')
        except Exception as e:
            print(f'    ✗ Parity plot failed: {e}')

        # 候補選択
        print(f'[{iteration}.3] Selecting {config.batch_size} candidates...')
        selected = pipeline.select_candidates(config.batch_size)
        print(f'    Selected: {selected}')

        # 実際のドッキングを実行
        print(f'\n[{iteration}.4] Running REAL docking for {len(selected)} sequences...')
        print('-'*70)

        docking_results = []
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)

        for i, seq in enumerate(selected, 1):
            print(f'\n  [{i}/{len(selected)}] Docking: {seq}')

            job_dir = os.path.join(work_dir, seq.lower())
            os.makedirs(job_dir, exist_ok=True)

            # ADCPコマンド（micromamba経由）
            cmd_parts = [
                'micromamba run -n adcpsuite adcp -O',
                f'-T {receptor_file}',
                f'-s "{seq}"',
                f'-o result_{seq.lower()}',
                f'-N {args.runs}',
                f'-n {args.evals}',
                '-cyc'
            ]

            # OpenMM rescoring for better accuracy (production mode)
            if args.omm_rescoring:
                cmd_parts.append('-nmin 5')  # Minimize top 5 poses with OpenMM

            cmd_parts.append(f'-w {job_dir}')
            cmd = ' '.join(cmd_parts)

            if args.verbose:
                print(f'      Command: {cmd}')
            else:
                print(f'      Command: {cmd[:80]}...')

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout
                )

                if result.returncode == 0:
                    # 結果をパース
                    dlg_file = os.path.join(job_dir, f"result_{seq.lower()}_summary.dlg")

                    if os.path.exists(dlg_file):
                        # affinityを抽出
                        with open(dlg_file, 'r') as f:
                            content = f.read()
                            import re
                            affinity_match = re.search(r'^\s+1\s+([-\d\.]+)', content, re.MULTILINE)
                            if affinity_match:
                                affinity = float(affinity_match.group(1))
                                print(f'      ✓ Success: {affinity:.2f} kcal/mol')
                                docking_results.append((seq, affinity))
                            else:
                                print(f'      ✗ Could not parse affinity')
                                docking_results.append((seq, 0.0))
                    else:
                        print(f'      ✗ DLG file not found')
                        docking_results.append((seq, 0.0))
                else:
                    print(f'      ✗ ADCP failed: {result.returncode}')
                    docking_results.append((seq, 0.0))

            except subprocess.TimeoutExpired:
                print(f'      ✗ Timeout')
                docking_results.append((seq, 0.0))
            except Exception as e:
                print(f'      ✗ Error: {e}')
                docking_results.append((seq, 0.0))

        # 結果を更新（更新前のベストを保存）
        print(f'\n[{iteration}.5] Updating pipeline with results...')
        previous_best = pipeline.state.best_aff
        pipeline.update(docking_results)

        # イテレーション結果のサマリー
        print(f'\n[{iteration}.6] Iteration {iteration} Summary:')
        print(f'    Total sequences evaluated: {len(pipeline.state.sequences)}')
        print(f'    Current best: {pipeline.state.best_seq} ({pipeline.state.best_aff:.2f} kcal/mol)')

        if pipeline.state.best_aff < previous_best:
            improvement = previous_best - pipeline.state.best_aff
            print(f'    ★ IMPROVED by {improvement:.2f} kcal/mol!')

        print(f'    New results this iteration:')
        for seq, aff in docking_results:
            marker = ' ★ NEW BEST!' if aff == pipeline.state.best_aff else ''
            print(f'      {seq}: {aff:.2f} kcal/mol{marker}')

        # イテレーション履歴を保存
        iteration_history.append({
            'iteration': iteration,
            'n_sequences': len(pipeline.state.sequences),
            'best_seq': pipeline.state.best_seq,
            'best_aff': pipeline.state.best_aff,
            'new_sequences': [seq for seq, _ in docking_results],
            'new_affinities': [aff for _, aff in docking_results]
        })

    # 最終パリティプロット作成（Train/Test split 80/20）
    print(f'\n{"="*70}')
    print('Creating Final Parity Plot (80/20 split)')
    print(f'{"="*70}')

    try:
        sequences = pipeline.state.sequences
        affinities = np.array(pipeline.state.affinities)

        if len(sequences) < 10:
            print(f'Not enough data for train/test split (n={len(sequences)})')
        else:
            # 80/20でTrain/Test分割
            seq_train, seq_test, y_train, y_test = train_test_split(
                sequences, affinities,
                test_size=0.2,
                random_state=config.seed
            )

            # Trainデータでモデルを学習
            pipeline.surrogate.fit(seq_train, y_train, verbose=False)

            # Train/Testデータで予測
            y_pred_train, y_std_train = pipeline.surrogate.predict(seq_train)
            y_pred_test, y_std_test = pipeline.surrogate.predict(seq_test)

            # 有効な予測のみ使用
            valid_train = ~np.isnan(y_pred_train) & ~np.isnan(y_train)
            valid_test = ~np.isnan(y_pred_test) & ~np.isnan(y_test)

            if valid_train.sum() > 0 and valid_test.sum() > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_plot_path = os.path.join(
                    config.output_dir,
                    'parity_plots',
                    f'parity_final_{timestamp}.png'
                )
                metrics = create_parity_plot(
                    y_train[valid_train], y_pred_train[valid_train],
                    y_test[valid_test], y_pred_test[valid_test],
                    y_std_train[valid_train] if y_std_train is not None else None,
                    y_std_test[valid_test] if y_std_test is not None else None,
                    output_path=final_plot_path,
                    iteration=None
                )
                print(f'\nFinal Model Metrics:')
                print(f'  Train: n={metrics["n_train"]}, R²={metrics["r2_train"]:.4f}, RMSE={metrics["rmse_train"]:.2f}')
                print(f'  Test:  n={metrics["n_test"]}, R²={metrics["r2_test"]:.4f}, RMSE={metrics["rmse_test"]:.2f}')
    except Exception as e:
        print(f'Final parity plot failed: {e}')

    # 最終結果表示
    print(f'\n{"="*70}')
    print('FINAL RESULTS')
    print(f'{"="*70}')

    print(f'\n[Overall Statistics]')
    print(f'  Total iterations: {config.n_iterations}')
    print(f'  Total sequences evaluated: {len(pipeline.state.sequences)}')
    print(f'  Initial best: {iteration_history[0]["best_seq"]} ({iteration_history[0]["best_aff"]:.2f} kcal/mol)')
    print(f'  Final best: {pipeline.state.best_seq} ({pipeline.state.best_aff:.2f} kcal/mol)')

    if pipeline.state.best_aff < iteration_history[0]["best_aff"]:
        improvement = iteration_history[0]["best_aff"] - pipeline.state.best_aff
        print(f'  Improvement: {improvement:.2f} kcal/mol ✓')
    else:
        print(f'  No improvement from initial best')

    print(f'\n[Iteration History]')
    for h in iteration_history:
        print(f'  Iter {h["iteration"]}: {h["n_sequences"]} seqs, Best: {h["best_aff"]:.2f} kcal/mol ({h["best_seq"]})')

    print(f'\n{"="*70}')
    print('✓ Active Learning completed!')
    print(f'{"="*70}')

    print(f'\nOutput files:')
    print(f'  Parity plots: {os.path.join(config.output_dir, "parity_plots")}/')

    return pipeline, iteration_history


if __name__ == '__main__':
    pipeline, history = main()
