import math
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax
)
from torch_geometric.nn.inits import zeros

class GATConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True,dropout: float = 0.0,
                 add_self_loops: bool = True,
                 bias: bool = True,attention_type: str = 'SD',**kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.alpha_ = None
        self.attention_type=attention_type
        self.lin = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, gx: Tensor, g_edge_index: Tensor) -> Tensor:
        N, H, C = x.size(0), self.heads, self.out_channels
        GN, GH, GC = gx.size(0), self.heads, self.out_channels
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            g_edge_index, _ = remove_self_loops(g_edge_index)
            g_edge_index, _ = add_self_loops(g_edge_index, num_nodes=GN)

        x = self.lin(x).view(-1, H, C)
        gx = self.lin(gx).view(-1, GH, GC)

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)
        out2 = self.propagate(edge_index, gx, size=None)

        #ここで各ノードで特徴量を連結する？


        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        return out
    
    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i)
        self.alpha_ = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    
    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                      num_nodes: Optional[int]) -> Tensor:
        if self.attention_type == "DP":
            alpha = (x_i * x_j).sum(-1)
        elif self.attention_type =="SD":
            alpha = (x_i * x_j).sum(-1)/math.sqrt(self.out_channels)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                )
    