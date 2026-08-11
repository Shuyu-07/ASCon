import torch
import torch.nn as nn
import torch.nn.functional as F


class DGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.out_features = out_features
        self.W_in = nn.Linear(in_features, out_features, bias=False)
        self.W_out = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=True)
        self.a_in = nn.Parameter(torch.empty(2 * out_features))
        self.a_out = nn.Parameter(torch.empty(2 * out_features))
        nn.init.normal_(self.a_in, mean=0.0, std=0.02)
        nn.init.normal_(self.a_out, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def _edge_attention(self, hidden, projection, attention, target, source):
        projected = projection(hidden)
        pair = torch.cat([projected[target], projected[source]], dim=-1)
        return self.activation((pair * attention).sum(dim=-1))

    @staticmethod
    def _group_softmax(scores, group_index, num_nodes):
        try:
            from torch_scatter import scatter_softmax
            return scatter_softmax(scores, group_index, dim=0, dim_size=num_nodes)
        except ImportError:
            maxima = torch.full((num_nodes,), float("-inf"), device=scores.device)
            maxima.scatter_reduce_(0, group_index, scores, reduce="amax", include_self=False)
            exponentials = torch.exp(scores - maxima[group_index])
            totals = torch.zeros(num_nodes, device=scores.device)
            totals.scatter_add_(0, group_index, exponentials)
            return exponentials / (totals[group_index] + 1e-8)

    def _aggregate(self, hidden, projection, attention, target, source, num_nodes):
        scores = self._edge_attention(hidden, projection, attention, target, source)
        weights = self.dropout(self._group_softmax(scores, target, num_nodes))
        messages = projection(hidden[source])
        result = torch.zeros(num_nodes, self.out_features, device=hidden.device)
        result.index_add_(0, target, weights.unsqueeze(-1) * messages)
        return result

    def forward(self, hidden, edge_index, num_nodes=None):
        num_nodes = hidden.size(0) if num_nodes is None else num_nodes
        source, target = edge_index if not isinstance(edge_index, torch.Tensor) else (edge_index[0], edge_index[1])
        source, target = source.to(hidden.device), target.to(hidden.device)
        self_state = self.W_self(hidden)
        if source.numel() == 0:
            return F.elu(self_state)
        incoming = self._aggregate(hidden, self.W_in, self.a_in, target, source, num_nodes)
        outgoing = self._aggregate(hidden, self.W_out, self.a_out, source, target, num_nodes)
        return F.elu(self_state + incoming + outgoing)


class DGATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_features)
        dimensions = (
            [(hidden_features, out_features)]
            if num_layers == 1
            else [(hidden_features, hidden_features)]
            + [(hidden_features, hidden_features)] * (num_layers - 2)
            + [(hidden_features, out_features)]
        )
        self.layers = nn.ModuleList([DGATLayer(source, target, dropout) for source, target in dimensions])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_features if index < num_layers - 1 else out_features)
            for index in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_index):
        hidden = F.elu(self.input_proj(hidden))
        for index, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer(hidden, edge_index)
            if hidden.shape == residual.shape:
                hidden = hidden + residual
            hidden = self.norms[index](hidden)
            if index < len(self.layers) - 1:
                hidden = self.dropout(hidden)
        return hidden
