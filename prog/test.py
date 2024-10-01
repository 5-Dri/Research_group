
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# カスタムGAT層：グラフ1とグラフ2を同時に処理
class DualGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(DualGATConv, self).__init__()
        # グラフ1とグラフ2の特徴量を処理するGAT層
        self.gat1 = GATConv(in_channels, out_channels, heads=heads)
        self.gat2 = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x1, edge_index1, x2, edge_index2, group_assignment):
        # グラフ1の特徴量にGATを適用
        x1_out = self.gat1(x1, edge_index1)

        # グラフ2の特徴量にGATを適用
        x2_out = self.gat2(x2, edge_index2)

        # グラフ1のノードが属するグループの特徴量を取得して統合
        x2_group = x2_out[group_assignment]  # グラフ1のノードに対応するグラフ2の特徴量

        # グラフ1とグラフ2の特徴量を統合（ここでは単純に足し合わせているが他の方法も可能）
        x1_combined = x1_out + x2_group

        return x1_combined, x2_out

# DualGATNetモデル
class DualGATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(DualGATNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DualGATConv(in_channels, out_channels))

    def forward(self, x1, edge_index1, x2, edge_index2, group_assignment):
        for layer in self.layers:
            x1, x2 = layer(x1, edge_index1, x2, edge_index2, group_assignment)
        return x1, x2

# データの準備
x1 = ...  # グラフ1のノード特徴量 (num_nodes1 x num_features)
edge_index1 = ...  # グラフ1のエッジリスト (2 x num_edges1)
x2 = ...  # グラフ2のノード特徴量 (num_groups x num_features)
edge_index2 = ...  # グラフ2のエッジリスト (2 x num_edges2)
group_assignment = ...  # グラフ1のノードが属するグループ (num_nodes1)

# モデルのインスタンス化と学習
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = x1.shape[1]
out_channels = 16  # 任意の出力特徴量次元
num_layers = 2  # GAT層の数

model = DualGATNet(in_channels, out_channels, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

x1 = x1.to(device)
edge_index1 = edge_index1.to(device)
x2 = x2.to(device)
edge_index2 = edge_index2.to(device)
group_assignment = torch.tensor(group_assignment, dtype=torch.long).to(device)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    x1_out, x2_out = model(x1, edge_index1, x2, edge_index2, group_assignment)
    loss = ...  # 任意の損失関数 (たとえばノード分類用の損失)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# モデルの評価や推論は適宜実行します。
