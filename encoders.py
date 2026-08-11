import torch
import torch.nn as nn
import torch.nn.functional as F


class StepEncoder(nn.Module):
    def __init__(self, text_dim, gru_hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(text_dim)
        self.bigru = nn.GRU(
            text_dim,
            gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_dim = gru_hidden_dim * 2
        self.output_proj = nn.Identity() if gru_dim == output_dim else nn.Linear(gru_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb):
        squeeze_batch = text_emb.dim() == 2
        if squeeze_batch:
            text_emb = text_emb.unsqueeze(0)
        hidden, _ = self.bigru(self.input_norm(text_emb.float()))
        hidden = self.output_norm(self.output_proj(self.dropout(hidden)))
        return hidden.squeeze(0) if squeeze_batch else hidden


class ContextAwareStepAgentPooling(nn.Module):
    def __init__(self, step_dim, role_dim, query_dim):
        super().__init__()
        self.query = nn.Linear(role_dim + step_dim, query_dim, bias=False)
        self.key = nn.Linear(step_dim, query_dim, bias=False)
        self.scale = query_dim ** -0.5

    def forward(self, H_s, role_embeddings, assignment_matrix):
        H_s = H_s.float()
        roles = role_embeddings.float()
        mask = assignment_matrix.float()
        local_mean = mask @ H_s / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        query = self.query(torch.cat([roles, local_mean], dim=-1))
        key = self.key(H_s)
        scores = (query @ key.T) * self.scale
        scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1) * mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return attention @ H_s, attention


class AgentEncoder(nn.Module):
    def __init__(self, step_dim, role_dim, hidden_dim, output_dim, query_dim, dropout=0.1):
        super().__init__()
        self.pooling = ContextAwareStepAgentPooling(step_dim, role_dim, query_dim)
        input_dim = role_dim + step_dim * 2
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, H_s, role_emb, assignment_matrix):
        aggregated, attention = self.pooling(H_s, role_emb, assignment_matrix)
        global_state = H_s.float().mean(dim=0, keepdim=True).expand(role_emb.size(0), -1)
        combined = torch.cat([role_emb.float(), aggregated, global_state], dim=-1)
        return self.encoder(combined), attention
