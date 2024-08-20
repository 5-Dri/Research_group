import random
import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
import numpy.linalg as LA




# すべてのノードが属するようにグループ分けを行う関数
def group_nodes_weighted_by_degree(G):
    nodes = set(G.nodes())
    degrees = dict(G.degree())
    # groups = {}
    # group_id = 0
    select_node = []
    groups = []

    while nodes:
        # group_id += 1
        # 重み付きランダムサンプリングによりノードを選択
        group = []
        weights = [degrees[node] for node in nodes]
        seed_node = random.choices(list(nodes), weights=weights, k=1)[0]
        select_node.append(seed_node)
        # 1ステップのノード（隣接ノード）を取得
        neighbors = list(G.neighbors(seed_node)) + [seed_node]
        # 新しいグループに追加
        for node in neighbors:
            if node in nodes:
                nodes.remove(node)
                group.append(node)
                # groups[node] = group_id
        groups.append(group)

    #subgraph = make_group_graph(G, select_node)


    #return groups, subgraph
    return groups , select_node

# 不要な関数
def filter_single_element_lists(list_of_lists):
    return [lst for lst in list_of_lists if len(lst) == 1]

def filter_adjacent_nodes(single_element_lists, G, target_nodes):
    single_nodes = [lst[0] for lst in single_element_lists if len(lst) == 1]

    #print("長さ１のグループ",len(single_nodes))

    adjacent_nodes_lists = []
    combined_lists = []

    for node in single_nodes:
        # for edge in graph:
        #     print(edge)
            # if node in edge:
            #     adjacent_nodes_lists.append(edge)]
        if node in target_nodes:
            target_nodes.remove(node)

        neighbors = list(G.neighbors(node))
        for lst in single_element_lists:
                if len(lst) > 1 and any(neighbor in lst for neighbor in neighbors):
                    adjacent_nodes_lists.append(lst)
                    combined_lists.append([node] + lst)

                    lst.append(node)

                    break  # 同じリストを複数回追加しないようにするため

    filtered_lists = [lst for lst in single_element_lists if len(lst) != 1]

    # print("隣接含むリスト",adjacent_nodes_lists)
    # print("合体",combined_lists)

    return filtered_lists, target_nodes

def compute_group_features_mean(groups, features):
    group_means = []

    for group in groups:
        group_features = torch.stack([features[int(node)] for node in group])
        group_mean = torch.mean(group_features, dim=0)
        group_means.append(group_mean)

    return torch.stack(group_means)




# グループごとのエッジインデックスをリスト型で作成する関数
#この関数は不要なはず。。
def create_group_edge_indices_list(G, partition):
    group_edges = {}
    for (u, v) in G.edges():
        group_u = partition[int(u)]
        group_v = partition[int(v)]
        if group_u == group_v:  # 同じグループのノード間のエッジのみ考慮
            if group_u not in group_edges:
                group_edges[group_u] = [[], []]  # 辺の始点リストと終点リストを初期化
            group_edges[group_u][0].append(u)
            group_edges[group_u][1].append(v)
    return group_edges


# PyTorch Geometricのデータ形式に変換する関数

def convert_to_pyg_data(group_edge_indices, data1):
    pyg_data_list = []
    for edges in group_edge_indices:
        edge_index = torch.tensor(edges, dtype=torch.long).to(device='cuda')
        data = Data(x=data1.x, edge_index=edge_index, y = data1.y, train_mask = data1.train_mask, val_mask = data1.val_mask, test_mask = data1.test_mask)
        pyg_data_list.append(data)
    return pyg_data_list


#グループをノードとするグラフ作成
def make_group_graph(G, target_nodes):


    subgraph = nx.Graph()
    subgraph.add_nodes_from(target_nodes)
    max_length = 3

    for node in target_nodes:
        for target_node in target_nodes:
             if node != target_node:  # 自分自身を除外
                try:
                    # path_length = nx.shortest_path_length(G, source=node, target=target_node)
                    all_paths = list(nx.all_simple_paths(G, source=node, target=target_node, cutoff=max_length))
                    # if path_length <= 3:
                    if len(all_paths) > 1:
                        subgraph.add_edge(node, target_node)
                except nx.NetworkXNoPath:
                    continue
    
    return subgraph

def get_eigen_zeros(G):
    c = 0
    laplacian = nx.laplacian_matrix(G)
    laplacian = laplacian.toarray()
    lams, p = LA.eigh(laplacian)

    for lam in lams:
        if lam < 0.01:
            c += 1

    return c


def graph_group(G, feature):
    groups, target_node = group_nodes_weighted_by_degree(G)

    final_groups, final_target_node = filter_adjacent_nodes(groups, G, target_node)

    group_feature = compute_group_features_mean(final_groups, feature)

    new_G = make_group_graph(G, final_target_node)

    return new_G, group_feature, final_target_node
