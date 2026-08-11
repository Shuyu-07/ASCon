import numpy as np
import torch

from .metrics import compute_aegis_fault_metrics_from_faulty_agents_pairs


def rank_result(logits):
    order = torch.argsort(logits, descending=True).tolist()
    return order[0], order


def evaluate_task1(model, samples, dataset_name, save_predictions=False):
    model.eval()
    device = next(model.parameters()).device
    records = []
    with torch.no_grad():
        for sample_id, sample in enumerate(samples):
            output = model(sample)
            step_y = sample["step_y"].to(device)
            agent_y = sample["agent_y"].to(device)
            gold_steps = torch.where(step_y > 0.5)[0].tolist()
            gold_agents = torch.where(agent_y > 0.5)[0].tolist()
            pred_step, step_order = rank_result(output["step_logits"])
            pred_agent, agent_order = rank_result(output["agent_logits"])
            row = {
                "dataset": dataset_name,
                "sample_id": sample_id,
                "file": sample.get("file", ""),
                "pred_step": pred_step,
                "pred_agent": pred_agent,
                "gold_steps": gold_steps,
                "gold_agents": gold_agents,
                "hit1": float(pred_step in gold_steps),
                "hit3": float(any(index in gold_steps for index in step_order[:3])),
                "agent_hit1": float(pred_agent in gold_agents),
                "agent_hit3": float(any(index in gold_agents for index in agent_order[:3])),
            }
            if save_predictions:
                row["step_logits"] = output["step_logits"].cpu().tolist()
                row["agent_logits"] = output["agent_logits"].cpu().tolist()
            records.append(row)
    metrics = {
        "num_samples": len(records),
        "step_micro_accuracy": float(np.mean([row["hit1"] for row in records])) if records else 0.0,
        "agent_micro_accuracy": float(np.mean([row["agent_hit1"] for row in records])) if records else 0.0,
    }
    return metrics, records


def add_binary_labels(samples):
    for sample in samples:
        sample["agent_error_binary"] = (sample["agent_error_type_ids"].long() > 0).float()
    return samples


AEGIS_ERROR_TYPES = [
    "benign",
    "FM-1.1",
    "FM-1.2",
    "FM-1.3",
    "FM-1.4",
    "FM-1.5",
    "FM-2.1",
    "FM-2.2",
    "FM-2.3",
    "FM-2.4",
    "FM-2.5",
    "FM-2.6",
    "FM-3.1",
    "FM-3.2",
    "FM-3.3",
]


def label_id_to_error_type(label_id):
    label_id = int(label_id)
    if 0 <= label_id < len(AEGIS_ERROR_TYPES):
        return AEGIS_ERROR_TYPES[label_id]
    return "benign"


def agent_faults_from_label_ids(agent_names, label_ids):
    faulty_agents = []
    for agent_name, label_id in zip(agent_names, label_ids):
        label_id = int(label_id)
        if label_id <= 0:
            continue
        faulty_agents.append(
            {
                "agent_name": str(agent_name),
                "error_type": label_id_to_error_type(label_id),
            }
        )
    return faulty_agents


def evaluate_task2(model, samples):
    model.eval()
    device = next(model.parameters()).device
    gold_faulty_agents_list = []
    pred_faulty_agents_list = []
    records = []
    with torch.no_grad():
        for sample in samples:
            gold = sample["agent_error_type_ids"].long().to(device)
            output = model(sample)
            binary_prob = torch.sigmoid(output["binary_logits"])
            binary_pred = (binary_prob >= 0.5).long()
            fault_type_pred = torch.argmax(output["fault_type_logits"], dim=-1) + 1
            combined_pred = torch.where(binary_pred > 0, fault_type_pred, torch.zeros_like(fault_type_pred))
            gold_list = gold.cpu().tolist()
            pred_list = combined_pred.cpu().tolist()
            agent_names = sample.get("agent_names", [])
            gold_faulty_agents = agent_faults_from_label_ids(agent_names, gold_list)
            pred_faulty_agents = agent_faults_from_label_ids(agent_names, pred_list)
            gold_faulty_agents_list.append(gold_faulty_agents)
            pred_faulty_agents_list.append(pred_faulty_agents)
            records.append(
                {
                    "file": sample.get("file", ""),
                    "gold_agent_error_type_ids": gold_list,
                    "pred_agent_error_type_ids": pred_list,
                    "pred_fault_prob": binary_prob.cpu().tolist(),
                    "gold_faulty_agents": gold_faulty_agents,
                    "pred_faulty_agents": pred_faulty_agents,
                }
            )
    metrics = compute_aegis_fault_metrics_from_faulty_agents_pairs(gold_faulty_agents_list, pred_faulty_agents_list)
    return {
        "faulty_agent": metrics["detect"],
        "error_mode": metrics["type"],
        "agent_error_pair": metrics["pair"],
        "records": records,
    }
