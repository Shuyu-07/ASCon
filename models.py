import torch
import torch.nn as nn

from .config import MODEL_CONFIG
from .dgat import DGATEncoder
from .encoders import AgentEncoder


def build_assignment_matrix(step_agent_ids, num_agents, num_steps, device):
    matrix = torch.zeros(num_agents, num_steps, device=device)
    matrix[step_agent_ids, torch.arange(num_steps, device=device)] = 1
    return matrix


def build_step_edges(num_steps, device):
    if num_steps <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    source = torch.arange(num_steps - 1, device=device)
    return torch.stack([source, source + 1], dim=0)


def mean_pool_steps_to_agents(step_features, step_agent_ids, num_agents):
    hidden_dim = step_features.size(-1)
    pooled = torch.zeros(num_agents, hidden_dim, device=step_features.device)
    counts = torch.zeros(num_agents, 1, device=step_features.device)
    pooled.index_add_(0, step_agent_ids, step_features)
    counts.index_add_(
        0,
        step_agent_ids,
        torch.ones(step_agent_ids.size(0), 1, device=step_features.device),
    )
    return pooled / counts.clamp(min=1.0)


class Task1ASCon(nn.Module):
    def __init__(self, step_text_dim, role_dim):
        super().__init__()
        cfg = MODEL_CONFIG
        step_dim = cfg["step_dim"]
        agent_dim = cfg["agent_dim"]
        dropout = cfg["dropout"]
        self.step_encoder = DGATEncoder(
            in_features=step_text_dim,
            hidden_features=cfg["dgat_hidden_dim"],
            out_features=step_dim,
            num_layers=cfg["dgat_layers"],
            dropout=dropout,
        )
        self.agent_encoder = AgentEncoder(
            step_dim,
            role_dim,
            cfg["agent_hidden_dim"],
            agent_dim,
            cfg["agent_query_dim"],
            dropout,
        )
        self.dgat = DGATEncoder(
            agent_dim,
            cfg["dgat_hidden_dim"],
            agent_dim,
            cfg["dgat_layers"],
            dropout,
        )
        self.agent_to_step = nn.Linear(agent_dim, step_dim)
        self.step_fusion = nn.Sequential(
            nn.LayerNorm(step_dim * 2),
            nn.Linear(step_dim * 2, step_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(step_dim),
        )
        self.step_classifier = nn.Sequential(
            nn.LayerNorm(step_dim),
            nn.Linear(step_dim, step_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(step_dim, 1),
        )
        self.agent_classifier = nn.Sequential(
            nn.LayerNorm(agent_dim),
            nn.Linear(agent_dim, agent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(agent_dim, 1),
        )

    def encode(self, sample):
        device = next(self.parameters()).device
        step_emb = torch.nan_to_num(sample["step_emb"].float().to(device))
        role_emb = torch.nan_to_num(sample["agent_emb"].float().to(device))
        step_agent_ids = sample["step_agent_ids"].long().to(device)
        agent_edges = sample["agent_edge_index"].long().to(device)
        step_edges = sample.get("step_edge_index")
        if step_edges is None:
            step_edges = build_step_edges(step_emb.size(0), device)
        else:
            step_edges = step_edges.long().to(device)
        assignment = build_assignment_matrix(step_agent_ids, role_emb.size(0), step_emb.size(0), device)
        H_s = self.step_encoder(step_emb, step_edges)
        H_a, attention = self.agent_encoder(H_s, role_emb, assignment)
        H_a = self.dgat(H_a, agent_edges)
        return H_s, H_a, step_agent_ids, attention

    def build_step_features(self, H_s, H_a, step_agent_ids):
        agent_for_step = self.agent_to_step(H_a[step_agent_ids])
        fused_input = torch.cat([H_s, H_s * agent_for_step], dim=-1)
        return self.step_fusion(fused_input), agent_for_step

    def forward(self, sample, return_debug=False):
        H_s, H_a, step_agent_ids, attention = self.encode(sample)
        step_features, agent_for_step = self.build_step_features(H_s, H_a, step_agent_ids)
        output = {
            "agent_logits": self.agent_classifier(H_a).squeeze(-1),
            "step_logits": self.step_classifier(step_features).squeeze(-1),
        }
        if return_debug:
            output["debug"] = {
                "H_s": H_s,
                "H_a": H_a,
                "agent_for_step": agent_for_step,
                "step_features": step_features,
                "pool_attention": attention,
            }
        return output


class Task2ASCon(nn.Module):

    def __init__(self, step_text_dim, role_dim, num_classes):
        super().__init__()
        cfg = MODEL_CONFIG
        step_dim = cfg["step_dim"]
        agent_dim = cfg["agent_dim"]
        dropout = cfg["dropout"]
        self.num_classes = num_classes
        self.step_encoder = DGATEncoder(
            in_features=step_text_dim,
            hidden_features=cfg["dgat_hidden_dim"],
            out_features=step_dim,
            num_layers=cfg["dgat_layers"],
            dropout=dropout,
        )
        self.agent_encoder = AgentEncoder(
            step_dim,
            role_dim,
            cfg["agent_hidden_dim"],
            agent_dim,
            cfg["agent_query_dim"],
            dropout,
        )
        self.agent_dgat = DGATEncoder(
            in_features=agent_dim,
            hidden_features=cfg["dgat_hidden_dim"],
            out_features=agent_dim,
            num_layers=cfg["dgat_layers"],
            dropout=dropout,
        )
        self.agent_to_step = nn.Linear(agent_dim, step_dim)
        self.step_feature_fusion = nn.Sequential(
            nn.LayerNorm(step_dim * 2),
            nn.Linear(step_dim * 2, step_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(step_dim),
        )
        self.agent_feature_fusion = nn.Sequential(
            nn.LayerNorm(agent_dim + step_dim),
            nn.Linear(agent_dim + step_dim, agent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(agent_dim),
        )
        self.binary_head = nn.Sequential(
            nn.LayerNorm(agent_dim),
            nn.Linear(agent_dim, agent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(agent_dim, 1),
        )
        self.fault_type_head = nn.Sequential(
            nn.LayerNorm(agent_dim),
            nn.Linear(agent_dim, agent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(agent_dim, num_classes - 1),
        )

    def encode_agent_features(self, sample):
        device = next(self.parameters()).device
        step_emb = torch.nan_to_num(sample["step_emb"].float().to(device))
        role_emb = torch.nan_to_num(sample["agent_emb"].float().to(device))
        step_agent_ids = sample["step_agent_ids"].long().to(device)
        agent_edges = sample["agent_edge_index"].long().to(device)
        step_edges = sample.get("step_edge_index")
        if step_edges is None:
            step_edges = build_step_edges(step_emb.size(0), device)
        else:
            step_edges = step_edges.long().to(device)
        assignment = build_assignment_matrix(step_agent_ids, role_emb.size(0), step_emb.size(0), device)

        H_s = self.step_encoder(step_emb, step_edges)
        H_a, _ = self.agent_encoder(H_s, role_emb, assignment)
        H_a = self.agent_dgat(H_a, agent_edges)
        agent_for_step = self.agent_to_step(H_a[step_agent_ids])
        step_features = self.step_feature_fusion(torch.cat([H_s, H_s * agent_for_step], dim=-1))
        pooled_step_features = mean_pool_steps_to_agents(step_features, step_agent_ids, H_a.size(0))
        return self.agent_feature_fusion(torch.cat([H_a, pooled_step_features], dim=-1))

    def forward(self, sample):
        agent_features = self.encode_agent_features(sample)
        return {
            "binary_logits": self.binary_head(agent_features).squeeze(-1),
            "fault_type_logits": self.fault_type_head(agent_features),
        }
