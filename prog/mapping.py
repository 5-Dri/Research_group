import torch

def convert_edge_index(group_nodes, edge_index):
    # 1. グループ番号へのマッピングを作成
    node_to_group = {}
    group_assignment = {}
    for group_idx, nodes in enumerate(group_nodes, start=0):  # グループ番号は1から始める
        group_assignment[group_idx] = nodes
        for node in nodes:
            node_to_group[node] = group_idx
    
    # 2. edge_indexをグループ番号に変換
    edge_index_mapped = []
    for src, dst in edge_index.t().tolist():  # edge_indexをリスト形式に変換して処理
        if src in node_to_group and dst in node_to_group:
            src_group = node_to_group[src]
            dst_group = node_to_group[dst]
            edge_index_mapped.append([src_group, dst_group])
    
    # edge_index_mappedをテンソルに変換
    edge_index_mapped = torch.tensor(edge_index_mapped, dtype=torch.long).t()
    
    return edge_index_mapped, group_assignment

# 入力データの例
group_nodes = [
    [1, 2, 3],          # グループ1のノード
    [4, 5, 6, 7, 8, 9, 10],  # グループ2のノード
    [11, 12, 13],       # グループ3のノード
    [14, 15, 16, 17],   # グループ4のノード
    [18, 19, 20]        # グループ5のノード
]

edge_index = torch.tensor([[1, 10, 10, 11, 15, 20], [10, 1, 11, 10, 20, 15]], dtype=torch.long)

# プログラム実行
edge_index_mapped, group_assignment = convert_edge_index(group_nodes, edge_index)

# 結果の表示
print("Mapped edge_index:")
print(edge_index_mapped)

print("\nGroup assignment:")
for group, nodes in group_assignment.items():
    print(f"Group {group}: {nodes}")
