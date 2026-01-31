#!/usr/bin/env python
"""
Active Learning Pipeline with Custom GNN for Cyclic Peptide Screening
======================================================================

【概要】
環状ペプチドのドッキングスコア（AutoDock CrankPep）を最小化するための
Active Learning / ベイズ最適化パイプライン

【このファイルの役割】
- GNN（グラフニューラルネットワーク）モデルの実装
- サロゲートモデル（代理モデル）の実装
- 獲得関数（次に試す配列を選ぶ関数）の実装

【主要なコンポーネント】

1. GNN層（CustomGNN）
   ペプチドの分子構造をグラフとして処理し、特徴量を抽出します。
   5種類のアーキテクチャを実装:
   - GIN: Graph Isomorphism Network
   - GCN: Graph Convolutional Network
   - GAT: Graph Attention Network
   - GraphSAGE: Sample and Aggregate
   - MPNN: Message Passing Neural Network

2. サロゲートモデル（SurrogateModel）
   GNN + ガウス過程（GP）を組み合わせたモデル。
   - GNN: ペプチドを固定長のベクトルに変換
   - GP: ベクトルから結合力を予測（不確実性付き）

3. 獲得関数
   次にどの配列をドッキングすべきかを決める関数:
   - EI (Expected Improvement): 期待される改善量
   - UCB (Upper Confidence Bound): 楽観的な予測
   - PI (Probability of Improvement): 改善する確率

【GNNの仕組み（初心者向け）】
ペプチドの分子構造を「グラフ」として表現します:
- ノード（点）: 各原子
- エッジ（線）: 原子間の結合

GNNは「メッセージパッシング」という仕組みで動作:
1. 各原子が隣接原子から情報を受け取る（メッセージ）
2. 受け取った情報を集約して自分の特徴を更新
3. これを複数回繰り返す
4. 全原子の情報を集約してペプチド全体の特徴を得る

【使用方法】
このファイルは直接実行せず、active_learning_from_lead.py から
インポートして使用します。

Usage:
    # 通常は active_learning_from_lead.py から呼び出される
    from active_learning_gnn import CustomGNN, SurrogateModel
"""

import os
import re
import json
import random
import warnings
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# Gaussian Process
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# RDKit
from rdkit import Chem

# Scipy
from scipy.stats import norm

# Sequence Models (Transformer, CNN, CatBoost)
from transformer_models import SequenceTransformerEncoder, GraphTransformerEncoder, CNN1DEncoder, CatBoostSurrogate

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration（設定）
# ============================================================================
# パイプライン全体の設定を管理するクラス。
# デフォルト値を変更することで、実験条件を調整できます。

@dataclass
class Config:
    """
    パイプライン設定

    【設定のカテゴリ】
    1. パス: データの入出力先
    2. GNN設定: ニューラルネットワークの構造
    3. GP設定: ガウス過程の学習パラメータ
    4. Active Learning設定: 最適化ループの設定
    5. ペプチド設定: 対象ペプチドの仕様
    """
    # パス
    result_dir: str = "/home/shizuku/In_silico_RaPID/Research_Linux/Research/result"
    output_dir: str = "/home/shizuku/In_silico_RaPID/Research_Linux/Research/al_output"

    # GNN設定
    # hidden_dim: 中間層のニューロン数（大きいほど表現力が高いが、過学習のリスクも増加）
    # output_dim: 出力ベクトルの次元数（GPへの入力）
    # num_layers: GNNの層数（深いほど遠くの原子の情報を取り込める）
    # dropout: 過学習を防ぐためにランダムにニューロンを無効化する割合
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.2

    # GP（ガウス過程）設定
    # training_iterations: GPの最適化イテレーション数
    # learning_rate: 学習率（大きいほど学習が速いが不安定になりやすい）
    gp_training_iterations: int = 50
    gp_learning_rate: float = 0.1

    # Active Learning設定
    # n_iterations: 最適化ループの回数
    # batch_size: 1イテレーションで評価する配列数
    # acquisition: 獲得関数の種類（EI=期待改善量、UCB=信頼上限、PI=改善確率）
    # ucb_beta: UCBの探索-活用バランスパラメータ（大きいほど探索重視）
    n_iterations: int = 20
    batch_size: int = 5
    acquisition: str = "EI"  # EI, UCB, PI
    ucb_beta: float = 2.0

    # ペプチド設定
    peptide_length: int = 13  # 環状ペプチドの長さ（アミノ酸数）
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"  # 使用可能な20種類のアミノ酸

    # その他
    seed: int = 42  # 乱数シード（再現性のため）
    device: str = "cpu"  # 計算デバイス（"cuda"でGPU使用）


# ============================================================================
# Amino Acid Properties（アミノ酸の物理化学的特性）
# ============================================================================
# 各アミノ酸には固有の物理化学的特性があります。
# これらの特性は、ペプチドの機能や受容体との相互作用に影響します。
#
# 【各特性の意味】
# - hydrophobicity（疎水性）: 水を嫌う性質。正の値が疎水的、負の値が親水的
# - charge（電荷）: +1（正電荷）, -1（負電荷）, 0（中性）
# - polar（極性）: 1（極性あり）, 0（非極性）
# - aromatic（芳香族）: 1（芳香環あり）, 0（なし）
# - mass（分子量）: アミノ酸の重さ（単位: Da）

AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 89.1},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 121.2},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polar': 1, 'aromatic': 0, 'mass': 133.1},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polar': 1, 'aromatic': 0, 'mass': 147.1},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'polar': 0, 'aromatic': 1, 'mass': 165.2},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 75.1},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'polar': 1, 'aromatic': 1, 'mass': 155.2},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 131.2},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'polar': 1, 'aromatic': 0, 'mass': 146.2},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 131.2},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 149.2},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'polar': 1, 'aromatic': 0, 'mass': 132.1},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 115.1},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polar': 1, 'aromatic': 0, 'mass': 146.2},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'polar': 1, 'aromatic': 0, 'mass': 174.2},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'polar': 1, 'aromatic': 0, 'mass': 105.1},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'polar': 1, 'aromatic': 0, 'mass': 119.1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'polar': 0, 'aromatic': 0, 'mass': 117.1},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'polar': 0, 'aromatic': 1, 'mass': 204.2},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polar': 1, 'aromatic': 1, 'mass': 181.2},
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ============================================================================
# Peptide to Graph Conversion（ペプチドからグラフへの変換）
# ============================================================================
# ペプチド配列（文字列）を、GNNで処理可能なグラフデータに変換します。
#
# 【変換の流れ】
# 1. アミノ酸配列（例: "HTIH..."）を受け取る
# 2. RDKitで分子構造を生成
# 3. 環状化（N末端とC末端を結合）
# 4. 各原子の特徴量を計算（ノード特徴）
# 5. 原子間の結合をエッジとして抽出
#
# 【グラフの構造】
# - ノード: 各原子（C, N, O, H など）
# - エッジ: 原子間の化学結合
# - ノード特徴: 原子番号、結合数、電荷、混成軌道、芳香族性など

def sequence_to_graph(sequence: str, cyclic: bool = True) -> Optional[Data]:
    """
    ペプチド配列をグラフデータに変換（RDKitを使用）

    【この関数の役割】
    文字列のペプチド配列を、GNNが処理できるグラフ形式に変換します。
    RDKitを使って分子構造を生成し、環状化も行います。

    【環状化とは？】
    環状ペプチドは、N末端（アミノ基 NH2）とC末端（カルボキシ基 COOH）が
    結合して環を形成しています。この結合を作成します。

    Parameters
    ----------
    sequence : str
        アミノ酸配列（例: "HTIHSWQMHFKIN"）
        大文字でも小文字でもOK
    cyclic : bool
        環状ペプチドとして処理するか（デフォルト: True）

    Returns
    -------
    Data
        torch_geometricのDataオブジェクト
        - x: ノード特徴量 [num_atoms, 6]
        - edge_index: エッジインデックス [2, num_edges]
    """
    try:
        seq = sequence.upper().strip()

        # RDKitで分子を作成
        mol = Chem.MolFromSequence(seq)
        if mol is None:
            return None

        # 環状化
        if cyclic:
            rwmol = Chem.RWMol(mol)
            n_matches = rwmol.GetSubstructMatches(Chem.MolFromSmarts("[NH2,NH3]"))
            c_matches = rwmol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[OH]"))

            if n_matches and c_matches:
                n_atom_idx = n_matches[0][0]
                c_term_indices = c_matches[-1]
                c_atom_idx = -1
                leaving_o_idx = -1

                for idx in c_term_indices:
                    atom = rwmol.GetAtomWithIdx(idx)
                    if atom.GetSymbol() == 'C':
                        for bond in atom.GetBonds():
                            neighbor = bond.GetOtherAtom(atom)
                            if neighbor.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.SINGLE:
                                c_atom_idx = idx
                                leaving_o_idx = neighbor.GetIdx()
                                break
                    if c_atom_idx != -1:
                        break

                if c_atom_idx != -1 and leaving_o_idx != -1:
                    rwmol.RemoveAtom(leaving_o_idx)
                    rwmol.AddBond(n_atom_idx, c_atom_idx, Chem.BondType.SINGLE)
                    mol = rwmol.GetMol()
                    Chem.SanitizeMol(mol)

        # ノード特徴量
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 20.0,  # 正規化
                atom.GetDegree() / 4.0,
                atom.GetFormalCharge() / 2.0,
                int(atom.GetHybridization()) / 6.0,
                float(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons() / 2.0,
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)

        # エッジ
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    except Exception:
        return None


def sequence_to_features(sequence: str) -> Optional[torch.Tensor]:
    """
    ペプチド配列を特徴量ベクトルに変換（RDKitを使わない簡易版）

    One-hot + 物理化学的特性
    """
    seq = sequence.upper().strip()

    features = []
    for aa in seq:
        if aa not in AA_TO_IDX:
            return None

        # One-hot
        one_hot = [0.0] * 20
        one_hot[AA_TO_IDX[aa]] = 1.0

        # 物理化学的特性
        props = AA_PROPERTIES.get(aa, {})
        prop_features = [
            props.get('hydrophobicity', 0) / 5.0,
            props.get('charge', 0) / 2.0,
            float(props.get('polar', 0)),
            float(props.get('aromatic', 0)),
            props.get('mass', 100) / 200.0,
        ]

        features.append(one_hot + prop_features)

    return torch.tensor(features, dtype=torch.float32)


# ============================================================================
# Custom GNN Implementation（GNNのスクラッチ実装）
# ============================================================================
# このセクションでは、5種類のGNNレイヤーを一から実装しています。
# ライブラリ（torch_geometric.nn.models）を使わず、メッセージパッシングの
# 仕組みを直接実装することで、カスタマイズ性と理解を深めています。
#
# 【メッセージパッシングの基本】
# 各ノード（原子）が隣接ノードから情報を「メッセージ」として受け取り、
# 自身の特徴を更新します。これを複数回繰り返すことで、
# 遠くのノードの情報も取り込めます。
#
# 【5種類のGNNレイヤー】
# 1. GCN (GraphConvLayer): 最もシンプル。隣接ノードの平均を取る
# 2. GIN (GINConvLayer): より表現力が高い。MLPで変換
# 3. GAT (GATConvLayer): 重要な隣接ノードに注目するアテンション機構
# 4. GraphSAGE (GraphSAGEConvLayer): 自己と隣接を別々に処理して結合
# 5. MPNN (MPNNConvLayer): 一般的なメッセージと更新関数

class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer（グラフ畳み込み層）

    【アルゴリズム】
    h_i' = σ(W * Σ_j h_j / deg(i))

    各ノードは隣接ノードの特徴を次数で割って平均し、
    線形変換（W）を適用します。

    【直感的な説明】
    「自分の隣人たちの意見の平均を聞いて、自分の意見を更新する」
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [num_nodes, in_channels]
            ノード特徴量
        edge_index : torch.Tensor [2, num_edges]
            エッジインデックス

        Returns
        -------
        torch.Tensor [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            # エッジがない場合
            return self.linear(x)

        # 次数の計算
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = deg.clamp(min=1)  # ゼロ除算防止

        # メッセージ集約
        # 隣接ノードの特徴量を平均
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
        out = out / deg.unsqueeze(1)

        # 自己ループ + 線形変換
        out = out + x
        out = self.linear(out)

        return out


class GINConvLayer(nn.Module):
    """
    Graph Isomorphism Network Layer（グラフ同型ネットワーク層）

    【アルゴリズム】
    h_i' = MLP((1 + ε) * h_i + Σ_j h_j)

    自分自身の特徴と隣接ノードの特徴の合計を、
    MLP（多層パーセプトロン）で変換します。

    【GINの特徴】
    - εパラメータで自己ループの重みを学習
    - GCNより表現力が高い（グラフの構造をより識別できる）
    - グラフの同型性テストに相当する識別能力

    【直感的な説明】
    「自分の意見を少し強調しつつ、隣人全員の意見を足し合わせる」
    """

    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor([eps]))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            out = (1 + self.eps) * x
        else:
            row, col = edge_index

            # 隣接ノードの特徴量を集約
            aggr = torch.zeros(num_nodes, x.size(1), device=x.device)
            aggr.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])

            out = (1 + self.eps) * x + aggr

        # BatchNorm用にサイズチェック
        if out.size(0) > 1:
            out = self.mlp(out)
        else:
            # バッチサイズ1の場合、BatchNormをスキップ
            out = self.mlp[0](out)  # Linear
            out = F.relu(out)
            out = self.mlp[3](out)  # Linear

        return out


class GATConvLayer(nn.Module):
    """
    Graph Attention Network Layer（グラフアテンション層）

    【アルゴリズム】
    h_i' = σ(Σ_j α_ij * W * h_j)
    α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))

    隣接ノードからの情報を、「アテンションスコア」で重み付けして集約します。
    重要な隣接ノードにより多くの注目を払います。

    【GATの特徴】
    - 各エッジに異なる重み（アテンション）を学習
    - マルチヘッドアテンション（複数の視点で注目）
    - ノードの重要度を動的に判断

    【直感的な説明】
    「重要な隣人の意見をより重視して聞く」
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout

        # 各ヘッドの線形変換
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        # アテンション係数
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # 線形変換 [num_nodes, heads * out_channels]
        x = self.linear(x)
        x = x.view(num_nodes, self.heads, self.out_channels)  # [N, H, C]

        if edge_index.size(1) == 0:
            return x.mean(dim=1)  # ヘッドを平均

        row, col = edge_index

        # アテンションスコア計算
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]

        # エッジごとのアテンションスコア
        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = self.leaky_relu(alpha)

        # Softmax（各ノードの入力エッジで正規化）
        alpha_max = torch.zeros(num_nodes, self.heads, device=x.device)
        alpha_max.scatter_reduce_(0, col.unsqueeze(1).expand(-1, self.heads), alpha, reduce='amax', include_self=False)
        alpha = torch.exp(alpha - alpha_max[col])

        alpha_sum = torch.zeros(num_nodes, self.heads, device=x.device)
        alpha_sum.scatter_add_(0, col.unsqueeze(1).expand(-1, self.heads), alpha)
        alpha = alpha / (alpha_sum[col] + 1e-8)

        # ドロップアウト
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # メッセージ集約
        out = torch.zeros(num_nodes, self.heads, self.out_channels, device=x.device)
        msg = x[row] * alpha.unsqueeze(-1)  # [E, H, C]
        out.scatter_add_(0, col.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.out_channels), msg)

        # ヘッドを平均
        return out.mean(dim=1)


class GraphSAGEConvLayer(nn.Module):
    """
    GraphSAGE Layer（スクラッチ実装）

    h_i' = σ(W * CONCAT(h_i, MEAN(h_j for j in N(i))))
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            # エッジがない場合、自己ループのみ
            return self.linear(torch.cat([x, x], dim=-1))

        row, col = edge_index

        # 隣接ノードの特徴量を集約（平均）
        aggr = torch.zeros(num_nodes, x.size(1), device=x.device)
        aggr.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])

        # 次数で正規化
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        aggr = aggr / deg

        # 自己特徴量と隣接特徴量を結合
        out = torch.cat([x, aggr], dim=-1)
        return self.linear(out)


class MPNNConvLayer(nn.Module):
    """
    Message Passing Neural Network Layer（スクラッチ実装）

    m_ij = MLP_msg(h_i, h_j)
    h_i' = MLP_update(h_i, Σ_j m_ij)
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 0):
        super().__init__()
        # メッセージ関数
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        # 更新関数
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.msg_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.update_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        if edge_index.size(1) == 0:
            # エッジがない場合
            zero_msg = torch.zeros(num_nodes, self.update_mlp[0].in_features - x.size(1), device=x.device)
            return self.update_mlp(torch.cat([x, zero_msg], dim=-1))

        row, col = edge_index

        # メッセージ計算
        msg_input = torch.cat([x[row], x[col]], dim=-1)
        msg = self.msg_mlp(msg_input)

        # メッセージ集約
        aggr = torch.zeros(num_nodes, msg.size(1), device=x.device)
        aggr.scatter_add_(0, col.unsqueeze(1).expand(-1, msg.size(1)), msg)

        # ノード更新
        out = torch.cat([x, aggr], dim=-1)
        return self.update_mlp(out)


class CustomGNN(nn.Module):
    """
    カスタムGNNモデル（スクラッチ実装）

    【このクラスの役割】
    複数のGNNレイヤーを積み重ねて、ペプチドのグラフから
    固定長のベクトル（埋め込み）を生成します。

    【アーキテクチャの選択】
    conv_typeパラメータで以下から選択可能:
    - GIN: 最も表現力が高い（デフォルト、推奨）
    - GCN: シンプルで高速
    - GAT: アテンション機構付き
    - GraphSAGE: サンプリングベース
    - MPNN: 汎用的なメッセージパッシング

    【処理の流れ】
    1. 入力: ペプチドのグラフ（原子=ノード、結合=エッジ）
    2. 複数のGNN層で特徴を抽出
    3. グローバルプーリングで全ノードを集約
    4. 出力: 固定長のベクトル（GPの入力になる）
    """

    # サポートするGNNタイプ
    SUPPORTED_TYPES = ["GIN", "GCN", "GAT", "GraphSAGE", "MPNN"]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "GIN"
    ):
        super().__init__()

        if conv_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"conv_type must be one of {self.SUPPORTED_TYPES}, got {conv_type}")

        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # 畳み込み層
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        def make_conv(in_ch, out_ch):
            """指定されたタイプの畳み込み層を作成"""
            if conv_type == "GIN":
                return GINConvLayer(in_ch, out_ch)
            elif conv_type == "GCN":
                return GraphConvLayer(in_ch, out_ch)
            elif conv_type == "GAT":
                return GATConvLayer(in_ch, out_ch, heads=4)
            elif conv_type == "GraphSAGE":
                return GraphSAGEConvLayer(in_ch, out_ch)
            elif conv_type == "MPNN":
                return MPNNConvLayer(in_ch, out_ch)

        # 最初の層
        self.convs.append(make_conv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 中間層
        for _ in range(num_layers - 2):
            self.convs.append(make_conv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 最後の層
        self.convs.append(make_conv(hidden_channels, hidden_channels))

        # 出力層
        self.fc = nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Data or Batch
            グラフデータ

        Returns
        -------
        torch.Tensor [batch_size, out_channels]
            グラフレベルの埋め込み
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # グラフ畳み込み
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if x.size(0) > 1:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最後の層
        x = self.convs[-1](x, edge_index)

        # グローバルプーリング（平均）
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        count = torch.zeros(batch_size, 1, device=x.device)

        out.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
        count.scatter_add_(0, batch.unsqueeze(1), torch.ones(x.size(0), 1, device=x.device))
        out = out / count.clamp(min=1)

        # 出力層
        out = self.fc(out)

        return out


# ============================================================================
# Gaussian Process Model（ガウス過程モデル）
# ============================================================================
# ガウス過程（GP）は、予測値だけでなく「不確実性」も出力できる回帰モデルです。
# 「この予測がどれくらい確かか」を知ることで、Active Learningで
# 探索（不確実な領域を調べる）と活用（良さそうな領域を深掘り）のバランスを取れます。
#
# 【GPの利点】
# - 予測値と不確実性（分散）の両方を出力
# - 少ないデータでも機能する
# - 過学習しにくい（ベイズ的アプローチ）
#
# 【GPの仕組み（簡略版）】
# 1. 訓練データ点間の「類似度」をカーネル関数で計算
# 2. 新しい点の予測は、訓練データとの類似度で重み付けした平均
# 3. 訓練データから遠い点ほど不確実性（分散）が大きくなる

class GPModel(ExactGP):
    """
    Gaussian Processモデル（gpytorch実装）

    【このクラスの役割】
    GNNで生成した埋め込みベクトルから、ドッキングスコア（affinity）を予測します。
    予測の平均と分散（不確実性）の両方を出力します。

    【使用するカーネル】
    - RBFKernel: 最も一般的なカーネル。滑らかな関数を仮定
    - ARD (Automatic Relevance Determination): 各次元の重要度を自動学習
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SurrogateModel:
    """
    GNN/Transformer + GP サロゲートモデル

    【サロゲートモデル（代理モデル）とは？】
    実際のドッキング計算の代わりに、結合力を高速に予測するモデルです。
    ドッキング計算は1配列あたり数分かかりますが、サロゲートモデルは数ミリ秒で予測できます。

    【このモデルの構造】
    1. エンコーダー（GNNまたはTransformer）
       - ペプチド配列/グラフを固定長のベクトルに変換
       - 「ペプチドの特徴」を数値で表現
    2. ガウス過程（GP）
       - ベクトルからaffinityを予測
       - 不確実性（どれくらい自信があるか）も出力

    【使い方】
    1. fit(): 既知のデータで学習
    2. predict(): 新しい配列のaffinityを予測（平均と標準偏差）
    """

    def __init__(self, encoder: nn.Module, config: Config, encoder_mode: str = "graph"):
        """
        サロゲートモデルを初期化

        Parameters
        ----------
        encoder : nn.Module
            エンコーダーモデル。以下のいずれか:
            - CustomGNN: グラフ入力（分子構造を直接処理）
            - SequenceTransformerEncoder: 配列入力（文字列を処理）
            - GraphTransformerEncoder: グラフ+Transformer
        config : Config
            設定オブジェクト（学習率、イテレーション数など）
        encoder_mode : str
            "graph": グラフ入力モード（GNN, GraphTransformer用）
            "sequence": 配列入力モード（SeqTransformer, CNN1D用）
        """
        self.encoder = encoder
        self.gnn = encoder  # 後方互換性のため
        self.config = config
        self.encoder_mode = encoder_mode
        self.gp = None
        self.likelihood = None
        self.X_train = None
        self.y_train = None
        self.fitted = False

    def fit(self, sequences: List[str], y_values: np.ndarray, verbose: bool = True):
        """モデル学習"""
        if self.encoder_mode == "sequence":
            # Sequence Transformer: 配列を直接入力
            valid_sequences = []
            valid_idx = []
            for i, seq in enumerate(sequences):
                if seq and len(seq) > 0:
                    valid_sequences.append(seq)
                    valid_idx.append(i)

            if not valid_sequences:
                raise ValueError("有効な配列がありません")

            # Transformerで埋め込み
            self.encoder.eval()
            with torch.no_grad():
                embeddings = self.encoder(valid_sequences)
        else:
            # Graph mode: グラフに変換してからエンコード
            graphs = []
            valid_idx = []

            for i, seq in enumerate(sequences):
                g = sequence_to_graph(seq)
                if g is not None:
                    graphs.append(g)
                    valid_idx.append(i)

            if not graphs:
                raise ValueError("有効なグラフがありません")

            # エンコーダーで埋め込み
            self.encoder.eval()
            with torch.no_grad():
                batch = Batch.from_data_list(graphs)
                embeddings = self.encoder(batch)

        self.X_train = embeddings
        self.y_train = torch.tensor(y_values[valid_idx], dtype=torch.float32)

        # GP初期化
        self.likelihood = GaussianLikelihood()
        self.gp = GPModel(self.X_train, self.y_train, self.likelihood)

        # 学習
        self.gp.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.config.gp_learning_rate)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)

        if verbose:
            print(f"  Training GP ({self.config.gp_training_iterations} iterations)...")

        for _ in range(self.config.gp_training_iterations):
            optimizer.zero_grad()
            output = self.gp(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()

        self.gp.eval()
        self.likelihood.eval()
        self.fitted = True

    def predict(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """予測（平均と標準偏差）"""
        if not self.fitted:
            raise ValueError("モデルが学習されていません")

        if self.encoder_mode == "sequence":
            # Sequence Transformer: 配列を直接入力
            valid_sequences = []
            valid_idx = []
            for i, seq in enumerate(sequences):
                if seq and len(seq) > 0:
                    valid_sequences.append(seq)
                    valid_idx.append(i)

            if not valid_sequences:
                return np.array([]), np.array([])

            self.encoder.eval()
            with torch.no_grad():
                X_test = self.encoder(valid_sequences)
        else:
            # Graph mode
            graphs = []
            valid_idx = []

            for i, seq in enumerate(sequences):
                g = sequence_to_graph(seq)
                if g is not None:
                    graphs.append(g)
                    valid_idx.append(i)

            if not graphs:
                return np.array([]), np.array([])

            self.encoder.eval()
            with torch.no_grad():
                batch = Batch.from_data_list(graphs)
                X_test = self.encoder(batch)

        self.gp.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp(X_test))
            mu = pred.mean.numpy()
            std = pred.stddev.numpy()

        return mu, std


# ============================================================================
# Acquisition Functions（獲得関数）
# ============================================================================
# 獲得関数は「次にどの候補を評価すべきか」を決めるための関数です。
# 予測値（mu）と不確実性（std）の両方を考慮して、候補にスコアを付けます。
#
# 【Active Learningのジレンマ：探索 vs 活用】
# - 探索（Exploration）: まだよく分かっていない領域を調べる → 不確実性が高い点を選ぶ
# - 活用（Exploitation）: 良さそうな領域を深掘りする → 予測値が良い点を選ぶ
#
# 獲得関数はこのバランスを自動的に取ってくれます。

def expected_improvement(mu: np.ndarray, std: np.ndarray, best_y: float) -> np.ndarray:
    """
    Expected Improvement（期待改善量）

    【この関数の意味】
    「この候補を評価したとき、現在のベストをどれくらい改善できそうか」
    の期待値を計算します。

    【計算式】
    EI = (best_y - mu) * Φ(z) + std * φ(z)
    z = (best_y - mu) / std
    Φ: 標準正規分布の累積分布関数
    φ: 標準正規分布の確率密度関数

    【特徴】
    - 最も広く使われる獲得関数
    - 探索と活用のバランスが良い
    - 予測値が良く、かつ不確実性も高い点を好む

    Parameters
    ----------
    mu : np.ndarray
        予測平均（各候補のaffinity予測値）
    std : np.ndarray
        予測標準偏差（各候補の不確実性）
    best_y : float
        現在のベストaffinity（最小値）

    Returns
    -------
    np.ndarray
        各候補のEIスコア（大きいほど有望）
    """
    with np.errstate(divide='ignore'):
        imp = best_y - mu  # 改善量（最小化なので best - mu）
        z = imp / (std + 1e-9)  # 標準化
        ei = imp * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-9] = 0  # 不確実性がない点はスキップ
    return ei


def upper_confidence_bound(mu: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound（信頼上限）

    【この関数の意味】
    「この候補は、楽観的に見てどれくらい良い可能性があるか」
    を計算します。

    【計算式】
    UCB = -(mu - beta * std)  # 最小化問題なので符号反転（LCB）

    【betaパラメータの意味】
    - beta が大きい: 探索重視（不確実な点を好む）
    - beta が小さい: 活用重視（予測が良い点を好む）
    - beta = 2.0 が一般的な値

    【特徴】
    - シンプルで計算が速い
    - betaでバランスを明示的に調整可能
    """
    return -(mu - beta * std)  # 最小化問題なのでLCBとして使用


def probability_of_improvement(mu: np.ndarray, std: np.ndarray, best_y: float) -> np.ndarray:
    """
    Probability of Improvement（改善確率）

    【この関数の意味】
    「この候補が現在のベストを改善する確率」を計算します。

    【計算式】
    PI = Φ((best_y - mu) / std)
    Φ: 標準正規分布の累積分布関数

    【特徴】
    - 直感的に理解しやすい（「改善する確率」）
    - 活用寄りの傾向（既に良い領域を集中的に探索）
    - EIより保守的な選択をする傾向
    """
    with np.errstate(divide='ignore'):
        z = (best_y - mu) / (std + 1e-9)
        pi = norm.cdf(z)
        pi[std < 1e-9] = 0
    return pi


# ============================================================================
# Data Loading
# ============================================================================

def load_docking_results(result_dir: str) -> pd.DataFrame:
    """ドッキング結果を読み込み"""
    if not os.path.exists(result_dir):
        return pd.DataFrame()

    data = []
    for seq in os.listdir(result_dir):
        seq_dir = os.path.join(result_dir, seq)
        if not os.path.isdir(seq_dir):
            continue

        dlg = os.path.join(seq_dir, f"result_{seq}_summary.dlg")
        if os.path.exists(dlg):
            with open(dlg, 'r') as f:
                for line in f:
                    m = re.search(r'^\s+1\s+([-\d\.]+)', line)
                    if m:
                        data.append({'sequence': seq, 'affinity': float(m.group(1))})
                        break

    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('affinity').reset_index(drop=True)
    return df


# ============================================================================
# Candidate Generation
# ============================================================================

def generate_random_sequences(n: int, length: int, existing: set = None) -> List[str]:
    """ランダム配列生成"""
    if existing is None:
        existing = set()

    aa = list("ACDEFGHIKLMNPQRSTVWY")
    candidates = []

    for _ in range(n * 10):
        seq = ''.join(random.choices(aa, k=length)).lower()
        if seq not in existing:
            candidates.append(seq)
            existing.add(seq)
        if len(candidates) >= n:
            break

    return candidates


def generate_mutants(base_seq: str, n_mut: int = 2, n_var: int = 10, existing: set = None) -> List[str]:
    """変異体生成"""
    if existing is None:
        existing = set()

    aa = list("ACDEFGHIKLMNPQRSTVWY")
    variants = []

    for _ in range(n_var * 10):
        seq_list = list(base_seq.upper())
        positions = random.sample(range(len(seq_list)), min(n_mut, len(seq_list)))
        for pos in positions:
            seq_list[pos] = random.choice(aa)

        var = ''.join(seq_list).lower()
        if var not in existing and var != base_seq.lower():
            variants.append(var)
            existing.add(var)

        if len(variants) >= n_var:
            break

    return variants


# ============================================================================
# Active Learning Pipeline（Active Learningパイプライン）
# ============================================================================
# このセクションは、Active Learning全体の流れを管理するクラスです。
# ※ 通常は active_learning_from_lead.py の ActiveLearningFromLead クラスを
#    使用しますが、このクラスはスタンドアロンでテスト実行できます。
#
# 【Active Learningの全体フロー】
# 1. 初期データを読み込み
# 2. サロゲートモデルを学習
# 3. 候補配列を生成
# 4. 獲得関数で有望な候補を選択
# 5. 選択した候補を評価（ドッキング）
# 6. データを更新して2に戻る
#
# これを繰り返すことで、少ない評価回数で最適な配列を見つけます。

@dataclass
class ALState:
    """
    Active Learning状態

    【このクラスの役割】
    Active Learningの現在の状態（どの配列を評価したか、ベストは何か）を管理します。
    """
    iteration: int = 0
    sequences: List[str] = field(default_factory=list)
    affinities: List[float] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    best_seq: str = ""
    best_aff: float = float('inf')
    history: List[Dict] = field(default_factory=list)


class ActiveLearningPipeline:
    """
    Active Learningパイプライン（スタンドアロン版）

    【このクラスの役割】
    Active Learning全体の流れを管理します。
    - 初期化: GNNとサロゲートモデルを作成
    - データ読み込み: 既存のドッキング結果を読み込み
    - 候補生成: ランダム配列と変異体を生成
    - 候補選択: 獲得関数で有望な配列を選択
    - 更新: 新しい評価結果でモデルを更新

    【使用方法】
    config = Config(n_iterations=10, batch_size=5)
    pipeline = ActiveLearningPipeline(config)
    history = pipeline.run()

    【注意】
    通常は active_learning_from_lead.py の方を使用してください。
    こちらはシミュレーション（実際のドッキングなし）でテスト用です。
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = ALState()
        self.gnn = None
        self.surrogate = None

        os.makedirs(config.output_dir, exist_ok=True)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    def initialize(self):
        """初期化"""
        # サンプルグラフで入力次元を取得
        sample = sequence_to_graph("ACDEFGHIKLMN")
        if sample is None:
            raise ValueError("サンプルグラフ生成失敗")

        in_dim = sample.x.shape[1]

        # GNN作成
        self.gnn = CustomGNN(
            in_channels=in_dim,
            hidden_channels=self.config.gnn_hidden_dim,
            out_channels=self.config.gnn_output_dim,
            num_layers=self.config.gnn_num_layers,
            dropout=self.config.gnn_dropout,
            conv_type="GIN"
        )

        # サロゲートモデル作成
        self.surrogate = SurrogateModel(self.gnn, self.config)

        print(f"GNN initialized: in={in_dim}, hidden={self.config.gnn_hidden_dim}, out={self.config.gnn_output_dim}")

    def load_data(self):
        """既存データ読み込み"""
        df = load_docking_results(self.config.result_dir)

        if df.empty:
            print("No existing data found")
            return

        self.state.sequences = df['sequence'].tolist()
        self.state.affinities = df['affinity'].tolist()

        best_idx = np.argmin(self.state.affinities)
        self.state.best_seq = self.state.sequences[best_idx]
        self.state.best_aff = self.state.affinities[best_idx]

        print(f"Loaded {len(df)} results")
        print(f"  Best: {self.state.best_seq} ({self.state.best_aff:.2f} kcal/mol)")

    def generate_candidates(self, n: int = 1000):
        """候補生成"""
        existing = set(self.state.sequences)

        # ランダム生成
        random_cands = generate_random_sequences(n // 2, self.config.peptide_length, existing)

        # 変異体生成（上位配列から）
        mutant_cands = []
        if self.state.sequences:
            sorted_idx = np.argsort(self.state.affinities)
            top_seqs = [self.state.sequences[i] for i in sorted_idx[:10]]
            for base in top_seqs:
                mutants = generate_mutants(base, n_mut=2, n_var=n // 20, existing=existing)
                mutant_cands.extend(mutants)

        self.state.candidates = random_cands + mutant_cands
        print(f"Generated {len(self.state.candidates)} candidates")

    def select_candidates(self, n: int) -> List[str]:
        """獲得関数で候補選択"""
        if not self.state.candidates:
            return []

        # 予測
        mu, std = self.surrogate.predict(self.state.candidates)

        if len(mu) == 0:
            return []

        # 獲得関数
        if self.config.acquisition == "EI":
            scores = expected_improvement(mu, std, self.state.best_aff)
        elif self.config.acquisition == "UCB":
            scores = upper_confidence_bound(mu, std, self.config.ucb_beta)
        elif self.config.acquisition == "PI":
            scores = probability_of_improvement(mu, std, self.state.best_aff)
        else:
            scores = expected_improvement(mu, std, self.state.best_aff)

        # 上位選択
        top_idx = np.argsort(scores)[-n:][::-1]
        selected = [self.state.candidates[i] for i in top_idx]

        # 候補から削除
        for s in selected:
            if s in self.state.candidates:
                self.state.candidates.remove(s)

        return selected

    def simulate_docking(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """ドッキングシミュレーション（プレースホルダー）"""
        results = []
        mu, std = self.surrogate.predict(sequences)

        for i, seq in enumerate(sequences):
            if i < len(mu):
                aff = mu[i] + np.random.normal(0, 0.5)
                results.append((seq, aff))

        return results

    def update(self, observations: List[Tuple[str, float]]):
        """観測データで更新"""
        for seq, aff in observations:
            self.state.sequences.append(seq)
            self.state.affinities.append(aff)

            if aff < self.state.best_aff:
                self.state.best_aff = aff
                self.state.best_seq = seq

    def run_iteration(self) -> Dict:
        """1イテレーション実行"""
        self.state.iteration += 1
        print(f"\n{'='*50}")
        print(f"Iteration {self.state.iteration}")
        print(f"{'='*50}")

        # サロゲートモデル学習
        print("Step 1: Training surrogate model...")
        self.surrogate.fit(self.state.sequences, np.array(self.state.affinities), verbose=False)

        # 候補選択
        print(f"Step 2: Selecting {self.config.batch_size} candidates...")
        selected = self.select_candidates(self.config.batch_size)

        if not selected:
            print("  Regenerating candidates...")
            self.generate_candidates()
            selected = self.select_candidates(self.config.batch_size)

        print(f"  Selected: {selected}")

        # ドッキング
        print("Step 3: Running docking...")
        observations = self.simulate_docking(selected)

        # 更新
        print("Step 4: Updating model...")
        self.update(observations)

        # 記録
        result = {
            'iteration': self.state.iteration,
            'n_obs': len(self.state.sequences),
            'best_aff': self.state.best_aff,
            'best_seq': self.state.best_seq,
            'selected': selected,
            'new_affs': [a for _, a in observations]
        }
        self.state.history.append(result)

        print(f"\nResults:")
        print(f"  Total: {len(self.state.sequences)}")
        print(f"  Best: {self.state.best_seq} ({self.state.best_aff:.2f} kcal/mol)")

        return result

    def run(self, n_iterations: int = None):
        """パイプライン実行"""
        if n_iterations is None:
            n_iterations = self.config.n_iterations

        print("="*50)
        print("Active Learning Pipeline (Custom GNN)")
        print("="*50)

        self.initialize()
        self.load_data()
        self.generate_candidates()

        for _ in range(n_iterations):
            self.run_iteration()

            if len(self.state.candidates) < self.config.batch_size * 2:
                self.generate_candidates()

        self.save_results()
        return self.state.history

    def save_results(self):
        """結果保存"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # numpy型をPython標準型に変換
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return obj

        # 履歴
        history_clean = convert(self.state.history)
        with open(os.path.join(self.config.output_dir, f"history_{ts}.json"), 'w') as f:
            json.dump(history_clean, f, indent=2)

        # 結果
        df = pd.DataFrame({
            'sequence': self.state.sequences,
            'affinity': self.state.affinities
        })
        df.to_csv(os.path.join(self.config.output_dir, f"results_{ts}.csv"), index=False)

        print(f"\nResults saved to {self.config.output_dir}")

    def get_top(self, n: int = 10) -> pd.DataFrame:
        """上位候補取得"""
        df = pd.DataFrame({
            'sequence': self.state.sequences,
            'affinity': self.state.affinities
        })
        return df.nsmallest(n, 'affinity')


# ============================================================================
# Visualization
# ============================================================================

def plot_progress(history: List[Dict], save_path: str = None):
    """最適化進捗プロット"""
    iters = [h['iteration'] for h in history]
    best = [h['best_aff'] for h in history]
    n_obs = [h['n_obs'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(iters, best, 'b-o', lw=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best Affinity (kcal/mol)')
    axes[0].set_title('Optimization Progress')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, n_obs, 'g-o', lw=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Number of Observations')
    axes[1].set_title('Data Accumulation')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    config = Config(
        n_iterations=10,
        batch_size=5,
        acquisition="EI",
        seed=42
    )

    pipeline = ActiveLearningPipeline(config)
    history = pipeline.run()

    print("\n" + "="*50)
    print("Top 10 Candidates")
    print("="*50)
    print(pipeline.get_top(10))

    plot_progress(history, os.path.join(config.output_dir, "progress.png"))

    return pipeline


if __name__ == "__main__":
    pipeline = main()
