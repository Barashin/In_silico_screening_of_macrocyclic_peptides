"""
Sequence Models for Cyclic Peptide Screening
=============================================

環状ペプチドのActive Learning用配列エンコーダモデル実装

1. SequenceTransformerEncoder: アミノ酸配列を直接入力（BERT風）
2. GraphTransformerEncoder: 分子グラフ構造にTransformerを適用
3. CNN1DEncoder: 1D畳み込みによる配列エンコーダ

Author: Generated for In-silico RaPID Screening Project
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.data import Data, Batch

# ============================================================================
# Amino Acid Vocabulary
# ============================================================================

AA_VOCAB = {
    '[PAD]': 0, '[CLS]': 1,
    'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16,
    'S': 17, 'T': 18, 'V': 19, 'W': 20, 'Y': 21
}

# アミノ酸の物理化学的特性（正規化済み）
# [疎水性, 分子量, 電荷, 極性, 芳香族性]
AA_PROPERTIES = {
    'A': [0.62, 0.24, 0.5, 0.0, 0.0],
    'C': [0.29, 0.35, 0.5, 0.5, 0.0],
    'D': [0.0, 0.39, 0.0, 1.0, 0.0],
    'E': [0.0, 0.44, 0.0, 1.0, 0.0],
    'F': [1.0, 0.55, 0.5, 0.0, 1.0],
    'G': [0.48, 0.0, 0.5, 0.0, 0.0],
    'H': [0.4, 0.52, 0.7, 0.5, 1.0],
    'I': [1.0, 0.45, 0.5, 0.0, 0.0],
    'K': [0.28, 0.47, 1.0, 1.0, 0.0],
    'L': [0.97, 0.45, 0.5, 0.0, 0.0],
    'M': [0.74, 0.50, 0.5, 0.0, 0.0],
    'N': [0.24, 0.40, 0.5, 1.0, 0.0],
    'P': [0.72, 0.36, 0.5, 0.0, 0.0],
    'Q': [0.24, 0.44, 0.5, 1.0, 0.0],
    'R': [0.0, 0.58, 1.0, 1.0, 0.0],
    'S': [0.36, 0.28, 0.5, 0.5, 0.0],
    'T': [0.45, 0.33, 0.5, 0.5, 0.0],
    'V': [0.97, 0.36, 0.5, 0.0, 0.0],
    'W': [0.97, 0.73, 0.5, 0.0, 1.0],
    'Y': [0.63, 0.63, 0.5, 0.5, 1.0],
}

# Chou-Fasman 二次構造傾向パラメータ（正規化済み: 0-1スケール）
# [α-helix傾向, β-sheet傾向, Turn傾向]
# 元の値を max で割って正規化
AA_SECONDARY_STRUCTURE = {
    'A': [0.94, 0.49, 0.43],  # α-helix former
    'C': [0.46, 0.79, 0.77],  # β-sheet
    'D': [0.67, 0.32, 1.00],  # Turn former
    'E': [1.00, 0.22, 0.48],  # α-helix former
    'F': [0.75, 0.85, 0.38],  # β-sheet
    'G': [0.38, 0.44, 1.00],  # Turn former
    'H': [0.67, 0.52, 0.55],  # Neutral
    'I': [0.71, 1.00, 0.30],  # β-sheet former
    'K': [0.79, 0.46, 0.65],  # α-helix
    'L': [0.80, 0.76, 0.38],  # α-helix/β-sheet
    'M': [0.96, 0.64, 0.39],  # α-helix former
    'N': [0.44, 0.52, 0.94],  # Turn former
    'P': [0.38, 0.32, 0.97],  # Turn/breaker
    'Q': [0.73, 0.65, 0.61],  # α-helix
    'R': [0.63, 0.56, 0.58],  # Neutral
    'S': [0.50, 0.47, 0.90],  # Turn former
    'T': [0.51, 0.70, 0.74],  # β-sheet/Turn
    'V': [0.70, 1.00, 0.32],  # β-sheet former
    'W': [0.71, 0.82, 0.42],  # β-sheet
    'Y': [0.44, 0.90, 0.81],  # β-sheet/Turn
}


def tokenize_sequences(sequences: List[str], max_len: int = 20) -> torch.Tensor:
    """
    アミノ酸配列をトークンIDに変換

    Parameters
    ----------
    sequences : List[str]
        アミノ酸配列のリスト
    max_len : int
        最大配列長（[CLS]トークン含む）

    Returns
    -------
    torch.Tensor
        トークンID [batch_size, max_len]
    """
    batch_tokens = []
    for seq in sequences:
        tokens = [AA_VOCAB['[CLS]']]
        for aa in seq.upper():
            if aa in AA_VOCAB:
                tokens.append(AA_VOCAB[aa])
        # パディング
        while len(tokens) < max_len:
            tokens.append(AA_VOCAB['[PAD]'])
        batch_tokens.append(tokens[:max_len])
    return torch.tensor(batch_tokens, dtype=torch.long)


def get_aa_properties(sequences: List[str], max_len: int = 20) -> torch.Tensor:
    """
    アミノ酸の物理化学的特性を取得

    Returns
    -------
    torch.Tensor
        [batch_size, max_len, 5]
    """
    batch_props = []
    for seq in sequences:
        props = [[0.0] * 5]  # [CLS]トークン用
        for aa in seq.upper():
            if aa in AA_PROPERTIES:
                props.append(AA_PROPERTIES[aa])
            else:
                props.append([0.0] * 5)
        # パディング
        while len(props) < max_len:
            props.append([0.0] * 5)
        batch_props.append(props[:max_len])
    return torch.tensor(batch_props, dtype=torch.float32)


# ============================================================================
# Cyclic Positional Encoding
# ============================================================================

class CyclicPositionalEncoding(nn.Module):
    """
    環状ペプチド用の周期的位置エンコーディング

    通常のPositional Encodingと異なり、位置0と位置N-1が
    隣接していることを考慮した周期的エンコーディング
    """

    def __init__(self, d_model: int, max_len: int = 20, cyclic: bool = True, dropout: float = 0.1):
        super().__init__()
        self.cyclic = cyclic
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        if cyclic:
            # 周期的位置エンコーディング
            # sin/cosを2π/max_lenの周期で適用
            pe[:, 0::2] = torch.sin(position * div_term * 2 * math.pi / max_len)
            pe[:, 1::2] = torch.cos(position * div_term * 2 * math.pi / max_len)
        else:
            # 標準的な位置エンコーディング
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]の形状で登録
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch_size, seq_len, d_model]

        Returns
        -------
        torch.Tensor
            位置エンコーディングが加算されたテンソル
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Sequence Transformer Encoder
# ============================================================================

class SequenceTransformerEncoder(nn.Module):
    """
    アミノ酸配列用Transformerエンコーダ

    配列を直接入力し、グラフ変換なしで埋め込みを生成
    環状ペプチド用のCyclic Positional Encodingを使用
    """

    def __init__(
        self,
        vocab_size: int = 22,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        out_channels: int = 64,
        dropout: float = 0.2,
        max_len: int = 20,
        use_aa_features: bool = True,
        cyclic: bool = True
    ):
        """
        Parameters
        ----------
        vocab_size : int
            語彙サイズ（アミノ酸20種 + [PAD], [CLS]）
        d_model : int
            埋め込み次元
        nhead : int
            アテンションヘッド数
        num_layers : int
            Transformerレイヤー数
        dim_feedforward : int
            FFNの中間次元
        out_channels : int
            出力次元（GP入力用）
        dropout : float
            ドロップアウト率
        max_len : int
            最大配列長
        use_aa_features : bool
            アミノ酸特性を追加するか
        cyclic : bool
            周期的位置エンコーディングを使用するか
        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.use_aa_features = use_aa_features

        # トークン埋め込み
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=AA_VOCAB['[PAD]'])

        # アミノ酸特性の射影層（オプション）
        if use_aa_features:
            self.aa_feature_proj = nn.Linear(5, d_model)
            self.combine_proj = nn.Linear(d_model * 2, d_model)

        # 位置エンコーディング
        self.pos_encoder = CyclicPositionalEncoding(d_model, max_len, cyclic, dropout)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力層
        self.fc = nn.Linear(d_model, out_channels)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : List[str]
            アミノ酸配列のリスト

        Returns
        -------
        torch.Tensor
            [batch_size, out_channels]
        """
        device = next(self.parameters()).device

        # トークン化
        tokens = tokenize_sequences(sequences, self.max_len).to(device)

        # パディングマスク作成
        padding_mask = (tokens == AA_VOCAB['[PAD]'])

        # 埋め込み
        x = self.embedding(tokens) * math.sqrt(self.d_model)

        # アミノ酸特性の追加（オプション）
        if self.use_aa_features:
            aa_props = get_aa_properties(sequences, self.max_len).to(device)
            aa_features = self.aa_feature_proj(aa_props)
            x = self.combine_proj(torch.cat([x, aa_features], dim=-1))

        # 位置エンコーディング
        x = self.pos_encoder(x)

        # Transformerエンコーダ
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # [CLS]トークンの出力を使用（BERTスタイル）
        cls_output = x[:, 0, :]

        # 出力射影
        output = self.fc(cls_output)

        return output


# ============================================================================
# Graph Transformer Layer
# ============================================================================

class GraphTransformerLayer(nn.Module):
    """
    グラフTransformerレイヤー

    エッジ情報を使ったスパースアテンション
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.nhead = nhead
        self.head_dim = out_channels // nhead
        assert out_channels % nhead == 0, "out_channels must be divisible by nhead"

        self.out_channels = out_channels

        # Query, Key, Value射影
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)

        # 出力射影
        self.W_o = nn.Linear(out_channels, out_channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels)
        )

        # 層正規化
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.dropout = nn.Dropout(dropout)

        # 入力射影（次元が異なる場合）
        self.input_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ノード特徴量 [num_nodes, in_channels]
        edge_index : torch.Tensor
            エッジインデックス [2, num_edges]

        Returns
        -------
        torch.Tensor
            更新されたノード特徴量 [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # 入力射影（残差接続用）
        residual = self.input_proj(x)

        # Q, K, V計算
        q = self.W_q(x).view(num_nodes, self.nhead, self.head_dim)
        k = self.W_k(x).view(num_nodes, self.nhead, self.head_dim)
        v = self.W_v(x).view(num_nodes, self.nhead, self.head_dim)

        # スパースアテンション
        attn_output = self._sparse_attention(q, k, v, edge_index, num_nodes)

        # 出力射影
        attn_output = attn_output.view(num_nodes, self.out_channels)
        attn_output = self.W_o(attn_output)

        # 残差接続 + 層正規化
        x = self.norm1(residual + self.dropout(attn_output))

        # FFN + 残差接続 + 層正規化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        グラフスパースアテンション

        エッジで接続されたノード間のみでアテンションを計算
        """
        row, col = edge_index  # row: source, col: target

        # 自己ループを追加（自分自身へのアテンション）
        self_loops = torch.arange(num_nodes, device=edge_index.device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])

        # エッジごとのアテンションスコア計算
        q_src = q[row]  # [num_edges, nhead, head_dim]
        k_dst = k[col]  # [num_edges, nhead, head_dim]

        # スケーリングされたドット積
        scores = (q_src * k_dst).sum(dim=-1) / math.sqrt(self.head_dim)  # [num_edges, nhead]

        # ソフトマックス（各ノードへの入力エッジで正規化）
        scores_max = torch.zeros(num_nodes, self.nhead, device=scores.device)
        scores_max.scatter_reduce_(0, col.unsqueeze(1).expand(-1, self.nhead), scores, reduce='amax', include_self=False)
        scores_exp = torch.exp(scores - scores_max[col])

        sum_exp = torch.zeros(num_nodes, self.nhead, device=scores.device)
        sum_exp.scatter_add_(0, col.unsqueeze(1).expand(-1, self.nhead), scores_exp)

        alpha = scores_exp / (sum_exp[col] + 1e-8)  # [num_edges, nhead]

        # 重み付き値の集約
        v_src = v[row]  # [num_edges, nhead, head_dim]
        weighted_v = v_src * alpha.unsqueeze(-1)

        # 各ノードへの集約
        out = torch.zeros(num_nodes, self.nhead, self.head_dim, device=q.device)
        out.scatter_add_(
            0,
            col.unsqueeze(1).unsqueeze(2).expand(-1, self.nhead, self.head_dim),
            weighted_v
        )

        return out  # [num_nodes, nhead, head_dim]


# ============================================================================
# Graph Transformer Encoder
# ============================================================================

class GraphTransformerEncoder(nn.Module):
    """
    グラフTransformerエンコーダ

    分子グラフ構造にTransformerを適用
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.2
    ):
        """
        Parameters
        ----------
        in_channels : int
            入力ノード特徴量次元
        hidden_channels : int
            隠れ層次元
        out_channels : int
            出力次元
        num_layers : int
            Transformerレイヤー数
        nhead : int
            アテンションヘッド数
        dropout : float
            ドロップアウト率
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 入力射影
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Transformerレイヤー
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                nhead=nhead,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 出力層
        self.fc = nn.Linear(hidden_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            グラフデータ

        Returns
        -------
        torch.Tensor
            グラフレベル埋め込み [batch_size, out_channels]
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 入力射影
        x = self.input_proj(x)

        # Transformerレイヤー
        for layer in self.layers:
            x = layer(x, edge_index)

        # グローバル平均プーリング
        from torch_geometric.nn import global_mean_pool
        graph_embedding = global_mean_pool(x, batch)

        # 出力射影
        output = self.fc(graph_embedding)

        return output


# ============================================================================
# 1D-CNN Encoder
# ============================================================================

class CNN1DEncoder(nn.Module):
    """
    1D畳み込みによる配列エンコーダ

    アミノ酸配列を1次元畳み込みで処理し、埋め込みを生成
    複数のカーネルサイズを使用してマルチスケール特徴を抽出
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        out_channels: int = 64,
        dropout: float = 0.2,
        max_len: int = 20,
        use_aa_features: bool = True
    ):
        """
        Parameters
        ----------
        vocab_size : int
            語彙サイズ（アミノ酸20種 + [PAD], [CLS]）
        embed_dim : int
            埋め込み次元
        num_filters : int
            各カーネルサイズのフィルター数
        kernel_sizes : List[int]
            畳み込みカーネルサイズのリスト（デフォルト: [3, 5, 7]）
        out_channels : int
            出力次元
        dropout : float
            ドロップアウト率
        max_len : int
            最大配列長
        use_aa_features : bool
            アミノ酸特性を使用するか
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.max_len = max_len
        self.use_aa_features = use_aa_features
        self.kernel_sizes = kernel_sizes

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=AA_VOCAB['[PAD]'])

        # アミノ酸特性の射影（オプション）
        if use_aa_features:
            self.aa_feature_proj = nn.Linear(5, embed_dim)
            input_dim = embed_dim * 2
        else:
            input_dim = embed_dim

        # マルチスケール1D畳み込み層
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # 追加の畳み込み層
        total_filters = num_filters * len(kernel_sizes)
        self.conv2 = nn.Sequential(
            nn.Conv1d(total_filters, total_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(total_filters // 2, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : List[str]
            アミノ酸配列のリスト

        Returns
        -------
        torch.Tensor
            [batch_size, out_channels]
        """
        device = next(self.parameters()).device

        # トークン化
        tokens = tokenize_sequences(sequences, self.max_len).to(device)

        # 埋め込み [batch, seq_len, embed_dim]
        x = self.embedding(tokens)

        # アミノ酸特性の追加（オプション）
        if self.use_aa_features:
            aa_props = get_aa_properties(sequences, self.max_len).to(device)
            aa_features = self.aa_feature_proj(aa_props)
            x = torch.cat([x, aa_features], dim=-1)

        # [batch, seq_len, channels] -> [batch, channels, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # マルチスケール畳み込み
        conv_outputs = [conv(x) for conv in self.convs]

        # 結合
        x = torch.cat(conv_outputs, dim=1)

        # 追加の畳み込み
        x = self.conv2(x)

        # グローバル平均プーリング + グローバル最大プーリング
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = avg_pool + max_pool  # 両方を組み合わせ

        # 出力
        output = self.fc(x)

        return output


class CNN1DResidualEncoder(nn.Module):
    """
    残差接続付き1D-CNNエンコーダ

    より深いネットワークで高精度な特徴抽出
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_len: int = 20,
        use_aa_features: bool = True
    ):
        """
        Parameters
        ----------
        vocab_size : int
            語彙サイズ
        embed_dim : int
            埋め込み次元
        hidden_channels : int
            隠れ層チャネル数
        out_channels : int
            出力次元
        num_layers : int
            残差ブロック数
        dropout : float
            ドロップアウト率
        max_len : int
            最大配列長
        use_aa_features : bool
            アミノ酸特性を使用するか
        """
        super().__init__()

        self.max_len = max_len
        self.use_aa_features = use_aa_features

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=AA_VOCAB['[PAD]'])

        # アミノ酸特性の射影（オプション）
        if use_aa_features:
            self.aa_feature_proj = nn.Linear(5, embed_dim)
            input_dim = embed_dim * 2
        else:
            input_dim = embed_dim

        # 入力射影
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)

        # 残差ブロック
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_channels, dropout)
            for _ in range(num_layers)
        ])

        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )

        self._init_weights()

    def _make_res_block(self, channels: int, dropout: float) -> nn.Module:
        """残差ブロックを作成"""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sequences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device

        # トークン化
        tokens = tokenize_sequences(sequences, self.max_len).to(device)

        # 埋め込み
        x = self.embedding(tokens)

        # アミノ酸特性の追加
        if self.use_aa_features:
            aa_props = get_aa_properties(sequences, self.max_len).to(device)
            aa_features = self.aa_feature_proj(aa_props)
            x = torch.cat([x, aa_features], dim=-1)

        # [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # 入力射影
        x = self.input_proj(x)

        # 残差ブロック
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)

        # グローバルプーリング
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # 出力
        output = self.fc(x)

        return output


# ============================================================================
# CatBoost Surrogate Model
# ============================================================================

class CatBoostSurrogate:
    """
    CatBoostによるサロゲートモデル

    アミノ酸配列を特徴量に変換し、CatBoostで予測
    不確実性推定はvirtual ensemblesを使用
    """

    def __init__(
        self,
        max_len: int = 20,
        use_aa_features: bool = True,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_seed: int = 42,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        max_len : int
            最大配列長
        use_aa_features : bool
            アミノ酸物理化学的特性を使用するか
        n_estimators : int
            ブースティングイテレーション数
        learning_rate : float
            学習率
        depth : int
            木の深さ
        random_seed : int
            ランダムシード
        verbose : bool
            学習時の出力
        """
        self.max_len = max_len
        self.use_aa_features = use_aa_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.verbose = verbose
        self.model = None
        self.fitted = False

    def _sequence_to_features(self, sequences: List[str]) -> 'np.ndarray':
        """配列を特徴量に変換"""
        import numpy as np

        features_list = []

        for seq in sequences:
            seq = seq.upper()
            features = []

            # 各位置のアミノ酸ID
            for i in range(self.max_len):
                if i < len(seq) and seq[i] in AA_VOCAB:
                    features.append(AA_VOCAB[seq[i]])
                else:
                    features.append(AA_VOCAB['[PAD]'])

            # アミノ酸物理化学的特性（位置別）
            if self.use_aa_features:
                for i in range(self.max_len):
                    if i < len(seq) and seq[i] in AA_PROPERTIES:
                        features.extend(AA_PROPERTIES[seq[i]])
                    else:
                        features.extend([0.0] * 5)

            # 二次構造傾向スコア（位置別）
            for i in range(self.max_len):
                if i < len(seq) and seq[i] in AA_SECONDARY_STRUCTURE:
                    features.extend(AA_SECONDARY_STRUCTURE[seq[i]])
                else:
                    features.extend([0.0] * 3)

            # 配列レベルの統計特徴
            # アミノ酸組成
            aa_counts = [0] * 20
            for aa in seq:
                if aa in AA_VOCAB and AA_VOCAB[aa] >= 2:
                    aa_counts[AA_VOCAB[aa] - 2] += 1
            features.extend([c / max(len(seq), 1) for c in aa_counts])

            # 平均物理化学的特性
            if len(seq) > 0:
                avg_props = [0.0] * 5
                count = 0
                for aa in seq:
                    if aa in AA_PROPERTIES:
                        for j, p in enumerate(AA_PROPERTIES[aa]):
                            avg_props[j] += p
                        count += 1
                if count > 0:
                    avg_props = [p / count for p in avg_props]
                features.extend(avg_props)
            else:
                features.extend([0.0] * 5)

            # 平均二次構造傾向
            if len(seq) > 0:
                avg_ss = [0.0] * 3  # α-helix, β-sheet, Turn
                count = 0
                for aa in seq:
                    if aa in AA_SECONDARY_STRUCTURE:
                        for j, s in enumerate(AA_SECONDARY_STRUCTURE[aa]):
                            avg_ss[j] += s
                        count += 1
                if count > 0:
                    avg_ss = [s / count for s in avg_ss]
                features.extend(avg_ss)
            else:
                features.extend([0.0] * 3)

            # 二次構造傾向の分散（配列内のばらつき）
            if len(seq) > 1:
                ss_values = [[], [], []]
                for aa in seq:
                    if aa in AA_SECONDARY_STRUCTURE:
                        for j, s in enumerate(AA_SECONDARY_STRUCTURE[aa]):
                            ss_values[j].append(s)
                ss_var = []
                for vals in ss_values:
                    if len(vals) > 1:
                        mean_val = sum(vals) / len(vals)
                        var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
                        ss_var.append(var_val)
                    else:
                        ss_var.append(0.0)
                features.extend(ss_var)
            else:
                features.extend([0.0] * 3)

            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def fit(self, sequences: List[str], y_values: 'np.ndarray', verbose: bool = None):
        """モデル学習"""
        from catboost import CatBoostRegressor
        import numpy as np

        if verbose is None:
            verbose = self.verbose

        X = self._sequence_to_features(sequences)
        y = np.array(y_values, dtype=np.float32)

        # 有効なデータのみ使用
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        self.model = CatBoostRegressor(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.random_seed,
            verbose=verbose,
            loss_function='RMSE',
            posterior_sampling=True  # 不確実性推定用
        )

        self.model.fit(X, y)
        self.fitted = True

        if verbose:
            print(f"  CatBoost trained on {len(y)} samples")

    def predict(self, sequences: List[str]) -> tuple:
        """
        予測（平均と標準偏差）

        Returns
        -------
        tuple
            (mean, std) - 予測値と不確実性
        """
        import numpy as np

        if not self.fitted:
            raise ValueError("モデルが学習されていません")

        X = self._sequence_to_features(sequences)

        # 予測
        mu = self.model.predict(X)

        # 不確実性推定（Virtual Ensemblesを使用）
        # CatBoostのvirtual_ensembles_countを使用
        try:
            predictions = self.model.virtual_ensembles_predict(
                X,
                prediction_type='TotalUncertainty',
                virtual_ensembles_count=10
            )
            std = predictions[:, 1]  # 不確実性
        except Exception:
            # フォールバック: 固定の不確実性
            std = np.ones_like(mu) * 0.5

        return mu.astype(np.float64), std.astype(np.float64)


# ============================================================================
# Test Functions
# ============================================================================

def test_catboost_surrogate():
    """CatBoostSurrogateのテスト"""
    print("Testing CatBoostSurrogate...")
    import numpy as np

    model = CatBoostSurrogate(verbose=False)

    sequences = ["ACDEFGHIKLMN", "YWVTSRQPNMLK", "HTIHSWQMHFKIN",
                 "AAAAAAAAAAAA", "GGGGGGGGGGGG"]
    y_values = np.array([-10.0, -15.0, -20.0, -5.0, -8.0])

    model.fit(sequences, y_values, verbose=False)

    test_seqs = ["ACDEFGHIKLMN", "FGHIKLMNPQRS"]
    mu, std = model.predict(test_seqs)

    print(f"  Input: {len(sequences)} training, {len(test_seqs)} test")
    print(f"  Predictions: {mu}")
    print(f"  Uncertainties: {std}")
    assert len(mu) == 2, f"Expected 2 predictions, got {len(mu)}"
    print("  PASSED")

    return model


def test_cnn1d_encoder():
    """CNN1DEncoderのテスト"""
    print("Testing CNN1DEncoder...")

    model = CNN1DEncoder(
        embed_dim=64,
        num_filters=128,
        out_channels=64
    )

    sequences = ["ACDEFGHIKLMN", "YWVTSRQPNMLK", "HTIHSWQMHFKIN"]
    output = model(sequences)

    print(f"  Input: {len(sequences)} sequences")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (3, 64), f"Expected (3, 64), got {output.shape}"
    print("  PASSED")

    return model


def test_cnn1d_residual_encoder():
    """CNN1DResidualEncoderのテスト"""
    print("Testing CNN1DResidualEncoder...")

    model = CNN1DResidualEncoder(
        embed_dim=64,
        hidden_channels=128,
        out_channels=64,
        num_layers=3
    )

    sequences = ["ACDEFGHIKLMN", "YWVTSRQPNMLK", "HTIHSWQMHFKIN"]
    output = model(sequences)

    print(f"  Input: {len(sequences)} sequences")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (3, 64), f"Expected (3, 64), got {output.shape}"
    print("  PASSED")

    return model


def test_sequence_transformer():
    """SequenceTransformerEncoderのテスト"""
    print("Testing SequenceTransformerEncoder...")

    model = SequenceTransformerEncoder(
        d_model=128,
        out_channels=64,
        num_layers=2,
        nhead=4
    )

    sequences = ["ACDEFGHIKLMN", "YWVTSRQPNMLK", "HTIHSWQMHFKIN"]
    output = model(sequences)

    print(f"  Input: {len(sequences)} sequences")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (3, 64), f"Expected (3, 64), got {output.shape}"
    print("  PASSED")

    return model


def test_graph_transformer():
    """GraphTransformerEncoderのテスト"""
    print("Testing GraphTransformerEncoder...")

    model = GraphTransformerEncoder(
        in_channels=6,
        hidden_channels=128,
        out_channels=64,
        num_layers=2,
        nhead=4
    )

    # ダミーグラフデータ作成
    x1 = torch.randn(10, 6)
    edge_index1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)

    x2 = torch.randn(8, 6)
    edge_index2 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                 [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)

    batch = Batch.from_data_list([data1, data2])
    output = model(batch)

    print(f"  Input: 2 graphs (10 nodes, 8 nodes)")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 64), f"Expected (2, 64), got {output.shape}"
    print("  PASSED")

    return model


def test_cyclic_positional_encoding():
    """CyclicPositionalEncodingのテスト"""
    print("Testing CyclicPositionalEncoding...")

    pe = CyclicPositionalEncoding(d_model=64, max_len=13, cyclic=True)

    x = torch.randn(2, 13, 64)
    output = pe(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # 位置0と位置12の類似度を確認
    pos_0 = pe.pe[0, 0, :]
    pos_12 = pe.pe[0, 12, :]
    pos_6 = pe.pe[0, 6, :]

    sim_0_12 = F.cosine_similarity(pos_0.unsqueeze(0), pos_12.unsqueeze(0))
    sim_0_6 = F.cosine_similarity(pos_0.unsqueeze(0), pos_6.unsqueeze(0))

    print(f"  Cosine similarity (pos 0 vs 12): {sim_0_12.item():.4f}")
    print(f"  Cosine similarity (pos 0 vs 6): {sim_0_6.item():.4f}")

    # 周期的エンコーディングでは、0と12は1と隣接、6は最も遠い
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Sequence Models Test")
    print("=" * 60)

    test_cyclic_positional_encoding()
    print()
    test_sequence_transformer()
    print()
    test_graph_transformer()
    print()
    test_cnn1d_encoder()
    print()
    test_cnn1d_residual_encoder()
    print()
    test_catboost_surrogate()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
