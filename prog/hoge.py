import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)
from layer import GATConv

# カスタムGAT層：グラフ1とグラフ2を同時に処理
class DualGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, att_type):
        super(DualGATConv, self).__init__()
        # グラフ1とグラフ2の特徴量を処理するGAT層
        self.gat1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=att_type)
        self.gat2 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, attention_type=att_type)

    def forward(self, x1, edge_index1, x2, edge_index2, group_assignment):
        # グラフ1の特徴量にGATを適用
        x1_out = self.gat1(x1, edge_index1)

        # グラフ2の特徴量にGATを適用
        x2_out = self.gat2(x2, edge_index2)

        # グラフ1のノードが属するグループの特徴量を取得して統合
        x2_group = x2_out[group_assignment]  # グラフ1のノードに対応するグラフ2の特徴量

        # グラフ1とグラフ2の特徴量を統合
        x1_combined = x1_out + x2_group

        return x1_combined, x2_out


# GATモデルにDualGATConvを適用
class GAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

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

    def forward(self, x1, edge_index1, x2, edge_index2, group_assignment):
        if self.cfg["num_layer"] != 1:
            # 初期層（グラフ1とグラフ2を処理）
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x1, x2 = self.inconv(x1, edge_index1, x2, edge_index2, group_assignment)
            x1 = self.in_norm(x1)
            x1 = F.elu(x1)

            # 中間層
        for mid_conv, mid_norm in zip(self.mid_convs, self.mid_norms):
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x1, x2 = mid_conv(x1, edge_index1, x2, edge_index2, group_assignment)
            x1 = mid_norm(x1)
            x1 = F.elu(x1)

        # 出力層
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1, x2 = self.outconv(x1, edge_index1, x2, edge_index2, group_assignment)
        x1 = self.out_norm(x1)

        return x1, x2

    def get_v_attention(self, edge_index,num_nodes,att):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E]
        v_att_l = []
        for v_index in range(num_nodes):
            att_neighbors = att[edge_index[1] == v_index, :].t()  # [heads, #neighbors]
            att_neighbors = att_neighbors.mean(dim=0)
            v_att_l.append(att_neighbors.to('cpu').detach().numpy().copy())
        return v_att_l    