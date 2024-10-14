import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
from layer import GATConv

def get_group_features(x2_out, group_assignment, node_indices):
    """
    各ノードに対応するグループの特徴量を取得する
    :param x2_out: グループノードの特徴量テンソル
    :param group_assignment: 各グループに属するノードのリスト
    :param node_indices: グラフ1のノードのインデックス
    :return: 各ノードに対応するグループの特徴量テンソル
    """
    node_to_group_features = []

    # ノードごとに、その属するグループの特徴量を取得
    print(type(node_indices))
    for node_idx in node_indices:
        # group_assignmentを検索してノードaが属するグループを見つける
        for group_idx, group in enumerate(group_assignment):
            if node_idx in group:
                # グループAの特徴量を取得
                group_features = x2_out[group_idx]
                node_to_group_features.append(group_features)
                break

    # テンソルに変換して返す
    return torch.stack(node_to_group_features, dim=0)

# カスタムGAT層：グラフ1とグラフ2を同時に処理
class DualGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, attention_type):
        super(DualGATConv, self).__init__()
        # グラフ1とグラフ2の特徴量を処理するGAT層
        # self.gat1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=attention_type, graph_type='origin')
        # self.gat2 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=attention_type, graph_type='group')
        self.gat1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=attention_type)
        self.gat2 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=attention_type)

    def forward(self, x, edge_index, x_g, edge_index_g, group_index):
        # グラフ1の特徴量にGATを適用
        # print("4", type(group_index))
        # print("x")
        # print(x.device)
        # print(type(x))
        # print(x.shape)
        # print("edge_index")
        # print(edge_index.device)
        # print(type(edge_index))
        # print(edge_index.shape)
        # print("g_x")
        # print(x_g.device)
        # print(type(x_g))
        # print(x_g.shape)
        # print("g_edge_index")
        # print(edge_index_g.device)
        # print(type(edge_index_g))
        # print(edge_index_g.shape)

        # assert edge_index.min() >= 0 and edge_index.max() < x.size(0), f"edge_index1 out of range: {edge_index.max()} >= {x.size(0)}"
        # assert edge_index_g.min() >= 0 and edge_index_g.max() < x_g.size(0), f"edge_index2 out of range: {edge_index_g.max()} >= {x_g.size(0)}"
        # assert group_index.min() >= 0, f"group_index contains negative values: {group_index.min()}"

        x_out = self.gat1(x, edge_index)

        # グラフ2の特徴量にGATを適用
        x_g_out = self.gat2(x_g, edge_index_g)

        # # グラフ1のノードが属するグループの特徴量を取得して統合
        # x_group = x_g_out[group_index]  # グラフ1のノードに対応するグラフ2の特徴量


        # # グループの特徴量と隣接グループの特徴量を集約
        # x_group = get_group_features(x_g_out, group_index, torch.arange(x.size(0)).tolist())

        # # グラフ1とグラフ2の特徴量を統合
        # out = x_out + x_group

        # # ノードごとの重要度計算を高速化
        # updated_node_features = torch.zeros_like(x_out)

        # # グループノードとその隣接ノードに基づいて特徴量を更新する処理をベクトル演算に変換
        # for group_idx, group in enumerate(group_index):
        #     group_feature = x_g_out[group_idx]  # グループノードの特徴量

        #     # 隣接グループノードの特徴を一度に取得
        #     adjacent_group_indices = edge_index_g[1][edge_index_g[0] == group_idx]
        #     adjacent_group_features = x_g_out[adjacent_group_indices].mean(dim=0)

        #     # グループ内の全ノードに対して一括処理
        #     node_features = x_out[group]  # グループ内のノード特徴量を取得

        #     # 内積計算をベクトル演算に変換
        #     importance_with_group = torch.einsum('nc,c->n', node_features, group_feature)  # (ノード数, 次元)と(次元)の内積
        #     importance_with_adjacent = torch.einsum('nc,c->n', node_features, adjacent_group_features)  # 同様に隣接グループとの内積

        #     # ノード特徴量の更新：重要度に基づく加算処理
        #     weighted_group_features = importance_with_group.unsqueeze(-1) * group_feature
        #     weighted_adjacent_features = importance_with_adjacent.unsqueeze(-1) * adjacent_group_features

        #     # 結果をノードの特徴量に追加
        #     updated_node_features[group] = node_features + weighted_group_features + weighted_adjacent_features

        # # 更新された特徴量でクラス予測
        # out = self.fc(updated_node_features)
        alpha1 = self.gat1.alpha_
        alpha2 = self.gat2.alpha_

        return x_out, x_g_out, alpha1, alpha2
    
    

# GATモデルにDualGATConvを適用
class GAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.data_index = None

        # Normalization layerの設定
        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid'] * cfg['n_head'])
            for _ in range(1, cfg["num_layer"] - 1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid'] * cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid'] * cfg['n_head'])
            for _ in range(1, cfg["num_layer"] - 1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid'] * cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1, cfg["num_layer"] - 1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()

        # レイヤーの定義
        if cfg["num_layer"] == 1:
            self.outconv = DualGATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
        else:
            self.inconv = DualGATConv(in_channels=cfg['n_feat'], out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            for _ in range(1, cfg["num_layer"] - 1):
                self.mid_convs.append(DualGATConv(in_channels=cfg['n_hid'] * cfg['n_head'], out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"]))
            self.outconv = DualGATConv(in_channels=cfg['n_hid'] * cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])

    def forward(self, x1, edge_index1, x2, edge_index2, group_index):
        if self.cfg["num_layer"] != 1:
            # 初期層（グラフ1とグラフ2を処理）
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            # print("3", type(group_index))

            x1, x2, alpha1, aplha2 = self.inconv(x1, edge_index1, x2, edge_index2, group_index)
            # print(type(x1))
            # print(x1.shape)

            # print(f"x1 shape: {x1.shape}")
            # print(f"LayerNorm normalized_shape: {self.in_norm.normalized_shape}")
            # print(f"x1 device: {x1.device}, LayerNorm device: {next(self.in_norm.parameters()).device}")

            # print(f"x1 device: {x1.device}, LayerNorm device: {next(self.in_norm.parameters()).device}")
            # print(f"x1 dtype: {x1.dtype}")
            # print(f"edge_index1 min: {edge_index1.min()}, max: {edge_index1.max()}, size: {x1.size(0)}")
            # print(f"edge_index2 min: {edge_index2.min()}, max: {edge_index2.max()}, size: {x2.size(0)}")
            # print(f"group_index min: {group_index.min()}, max: {group_index.max()}")

            x1 = self.in_norm(x1)
            x1 = F.elu(x1)

            # 中間層
        for mid_conv, mid_norm in zip(self.mid_convs, self.mid_norms):
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x1, x2, alpha1, aplha2  = mid_conv(x1, edge_index1, x2, edge_index2, group_index)
            x1 = mid_norm(x1)
            x1 = F.elu(x1)

        # 出力層
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1, x2, alpha1, aplha2  = self.outconv(x1, edge_index1, x2, edge_index2, group_index)
        x1 = self.out_norm(x1)

        return x1, [], alpha1


    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]
        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())
        return v_att_l    