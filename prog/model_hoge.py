import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
from layer import GATConv


class UnifiedGATModel(nn.Module):
    def __init__(self, cfg, group_nodes):
        super(UnifiedGATModel, self).__init__()
        self.cfg = cfg
        self.group_nodes = group_nodes

        # GATレイヤー定義
        self.gat_conv_full = GATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'], attention_type=cfg["att_type"])
        self.gat_conv_group = GATConv(cfg['n_group_feat'], cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'], attention_type=cfg["att_type"])
        
        # 出力層 (ノードのクラス予測用)
        self.fc = nn.Linear(cfg['n_hid'], cfg['n_class'])

    def forward(self, data_full, data_group):
        # 元のグラフでGAT畳み込み
        full_x = F.elu(self.gat_conv_full(data_full.x, data_full.edge_index))

        # グループグラフでGAT畳み込み
        group_x = F.elu(self.gat_conv_group(data_group.x, data_group.edge_index))

        # ノード特徴量の更新（重み付き畳み込み）
        updated_node_features = torch.zeros_like(full_x)
        
        for group_idx, group in enumerate(self.group_nodes):
            group_feature = group_x[group_idx]  # グループノードの特徴量

            # 隣接グループノードの特徴を取得
            adjacent_group_indices = data_group.edge_index[1][data_group.edge_index[0] == group_idx]
            adjacent_group_features = group_x[adjacent_group_indices]
            
            for node_idx in group:
                node_feature = full_x[node_idx]  # 元のノードの特徴量

                # グループノードとの重要度を計算（内積など）
                importance_with_group = torch.dot(node_feature, group_feature)
                importance_with_adjacent = torch.matmul(adjacent_group_features, node_feature.unsqueeze(-1)).mean()

                # 重要度を基にグループノードの特徴量を重み付けして畳み込み
                weighted_group_feature = importance_with_group * group_feature
                weighted_adjacent_feature = importance_with_adjacent * adjacent_group_features.mean(dim=0)

                # ノードの特徴量を更新
                updated_node_features[node_idx] = node_feature + weighted_group_feature + weighted_adjacent_feature

        # 更新された特徴量でクラス予測
        out = self.fc(updated_node_features)
        return out, updated_node_features