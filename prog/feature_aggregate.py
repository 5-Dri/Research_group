import torch
import torch.nn.functional as F


import torch

# ノードの特徴ベクトル（GのX）
X = torch.randn(20, 16)  # ノードが20個、各ノードの特徴次元が16次元の場合

# グループの割り当て
group_assignment = {
    0: [1, 2, 3],
    1: [4, 5, 6, 7, 8, 9, 10],
    2: [11, 12, 13],
    3: [14, 15, 16, 17],
    4: [18, 19, 20]
}

# G2のグループノード間のエッジインデックス（G2のE2）
E2 = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]])

# 各グループの特徴ベクトルX2を計算
group_features = {}
for group, nodes in group_assignment.items():
    node_indices = torch.tensor([n - 1 for n in nodes])  # ノードインデックスは0から始まるため
    group_features[group] = X[node_indices].mean(dim=0)  # グループ内のノードの特徴ベクトルを平均

# グループ特徴ベクトルX2のテンソル
X2 = torch.stack([group_features[group] for group in range(len(group_assignment))])

# ノードviが属するグループgiを取得
node_to_group = {}
for group, nodes in group_assignment.items():
    for node in nodes:
        node_to_group[node - 1] = group  # ノードインデックスは0ベースに調整



# ノードviに対するグループと隣接グループとの類似度を計算
for node_idx in range(X.size(0)):
    sim_group, sim_adj_group = aggregate_features(node_idx)
    print(f"Node {node_idx + 1}:")
    print(f"  Similarity with group: {sim_group.item():.4f}")
    print(f"  Similarity with adjacent group: {sim_adj_group.item():.4f}")


def aggregate_features_with_attention(node_to_group):
    for node_idx in range(X.size(0)):

        group_idx = node_to_group[node_idx]  # ノードが属するグループのインデックスを取得
        group_feat = X2[group_idx]  # giの特徴ベクトルを取得

        # 隣接グループの特徴ベクトルを取得
        adjacent_group_feats = []
        similarities_with_adjacent_groups = []
        for edge in E2:
            if edge[0] == group_idx:
                adj_group_feat = X2[edge[1]]
                adjacent_group_feats.append(adj_group_feat)
                # 類似度を計算
                similarities_with_adjacent_groups.append(torch.dot(X[node_idx], adj_group_feat))
            if edge[1] == group_idx:
                adj_group_feat = X2[edge[0]]
                adjacent_group_feats.append(adj_group_feat)
                # 類似度を計算
                similarities_with_adjacent_groups.append(torch.dot(X[node_idx], adj_group_feat))

        # ノードviの特徴ベクトルとグループの特徴ベクトルを内積で比較
        node_feat = X[node_idx]  # ノードviの特徴ベクトル
        similarity_with_group = torch.dot(node_feat, group_feat)  # giとの内積（類似度）

        # 類似度リストの作成: 自グループ + 隣接グループ
        all_similarities = [similarity_with_group] + similarities_with_adjacent_groups

        # 類似度をソフトマックスで確率に変換
        attention_weights = F.softmax(torch.tensor(all_similarities), dim=0)

        # それぞれの特徴ベクトルに確率をかけて足し合わせ
        weighted_group_feat = attention_weights[0] * group_feat  # 自グループの特徴に対する重み
        weighted_adj_feats = [attention_weights[i + 1] * adjacent_group_feats[i] for i in range(len(adjacent_group_feats))]

        # 全ての特徴ベクトルの加重和を計算
        final_feat = weighted_group_feat + sum(weighted_adj_feats)

        return final_feat, attention_weights
