import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, NNConv, GCNConv, GATConv, GATv2Conv, GINConv, TransformerConv
)


class EdgeFraudNetFlexible(nn.Module):
    def __init__(
        self,
        input_dim,
        edge_attr_dim,
        hidden_size=128,
        gnn_type="GINE",
        num_layers=2,
        dropout=0.0,
        heads=4
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.heads = heads

        # Input encoders
        self.node_encoder = nn.Linear(input_dim, hidden_size)
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_size)

        # Choose embedding dimension depending on the GNN type
        if gnn_type == "TransformerConv":
            self.embedding_dim = hidden_size * heads  # concat=True
        elif gnn_type == "GAT" or gnn_type == "GATv2":
            self.embedding_dim = hidden_size  # already hidden_size
        else:
            self.embedding_dim = hidden_size

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            if gnn_type == "GINE":
                self.convs.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU()
                        ),
                        edge_dim=hidden_size
                    )
                )
            elif gnn_type == "NNConv":
                edge_nn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * hidden_size)
                )
                self.convs.append(NNConv(hidden_size, hidden_size, edge_nn, aggr="mean"))
            elif gnn_type == "TransformerConv":
                self.convs.append(
                    TransformerConv(
                        hidden_size,
                        hidden_size,
                        edge_dim=hidden_size,
                        heads=heads,
                        concat=True  # Ensure output dim is hidden_size * heads
                    )
                )
            elif gnn_type == "GCN":
                self.convs.append(GCNConv(hidden_size, hidden_size))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GATv2":
                self.convs.append(GATv2Conv(hidden_size, hidden_size // heads, heads=heads))
            elif gnn_type == "GIN":
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU()
                        )
                    )
                )
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, edge_index, edge_attr, edge_label_index, edge_label_attr):
        x = F.leaky_relu(self.node_encoder(x))
        edge_attr = F.leaky_relu(self.edge_encoder(edge_attr))
        edge_label_attr = F.leaky_relu(self.edge_encoder(edge_label_attr))

        for conv in self.convs:
            if self.gnn_type in ["GINE", "NNConv", "TransformerConv"]:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h_src = x[edge_label_index[0]]
        h_dst = x[edge_label_index[1]]
        combined = torch.cat([h_src, h_dst, edge_label_attr], dim=1)
        out = self.mlp(combined)

        return out.view(-1), x
