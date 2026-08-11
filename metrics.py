import re

import numpy as np


def set_f1_metrics(true_labels_list, pred_labels_list):
    all_classes = set()
    for labels in true_labels_list:
        all_classes.update(labels)
    for labels in pred_labels_list:
        all_classes.update(labels)
    all_classes = list(all_classes)

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    per_class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in all_classes}

    for true_set, pred_set in zip(true_labels_list, pred_labels_list):
        tp_set = true_set.intersection(pred_set)
        fp_set = pred_set.difference(true_set)
        fn_set = true_set.difference(pred_set)
        micro_tp += len(tp_set)
        micro_fp += len(fp_set)
        micro_fn += len(fn_set)
        for cls in tp_set:
            per_class_stats[cls]["tp"] += 1
        for cls in fp_set:
            per_class_stats[cls]["fp"] += 1
        for cls in fn_set:
            per_class_stats[cls]["fn"] += 1

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_f1_scores = []
    for cls in all_classes:
        tp = per_class_stats[cls]["tp"]
        fp = per_class_stats[cls]["fp"]
        fn = per_class_stats[cls]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_f1_scores.append(f1)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(macro_f1_scores)) if macro_f1_scores else 0.0,
    }


def fault_sets_from_label_ids(label_ids):
    detect = {index for index, label in enumerate(label_ids) if int(label) > 0}
    fault_type = {int(label) for label in label_ids if int(label) > 0}
    pair = {(index, int(label)) for index, label in enumerate(label_ids) if int(label) > 0}
    return detect, fault_type, pair


def compute_aegis_fault_metrics_from_label_id_pairs(gold_sequences, pred_sequences):
    detect_gold = []
    detect_pred = []
    type_gold = []
    type_pred = []
    pair_gold = []
    pair_pred = []

    for gold_labels, pred_labels in zip(gold_sequences, pred_sequences):
        gold_detect, gold_type, gold_pair = fault_sets_from_label_ids(gold_labels)
        pred_detect, pred_type, pred_pair = fault_sets_from_label_ids(pred_labels)
        detect_gold.append(gold_detect)
        detect_pred.append(pred_detect)
        type_gold.append(gold_type)
        type_pred.append(pred_type)
        pair_gold.append(gold_pair)
        pair_pred.append(pred_pair)

    return {
        "detect": set_f1_metrics(detect_gold, detect_pred),
        "type": set_f1_metrics(type_gold, type_pred),
        "pair": set_f1_metrics(pair_gold, pair_pred),
    }


def normalize_agent_name(value):
    name = str(value or "").strip()
    if not name:
        return ""
    name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
    return re.sub(r"\s+", " ", name).casefold()


def normalize_error_type(value):
    return str(value or "").strip().upper()


def fault_sets_from_faulty_agents(faulty_agents, benign_label="benign"):
    pair = set()
    if not isinstance(faulty_agents, list):
        faulty_agents = []
    for item in faulty_agents:
        if not isinstance(item, dict):
            continue
        agent_name = normalize_agent_name(item.get("agent_name"))
        error_type = normalize_error_type(item.get("error_type"))
        if not agent_name or not error_type or error_type == benign_label.upper():
            continue
        pair.add((agent_name, error_type))
    detect = {agent_name for agent_name, _ in pair}
    fault_type = {error_type for _, error_type in pair}
    return detect, fault_type, pair


def compute_aegis_fault_metrics_from_faulty_agents_pairs(gold_faulty_agents_list, pred_faulty_agents_list):
    detect_gold = []
    detect_pred = []
    type_gold = []
    type_pred = []
    pair_gold = []
    pair_pred = []

    for gold_faulty_agents, pred_faulty_agents in zip(gold_faulty_agents_list, pred_faulty_agents_list):
        gold_detect, gold_type, gold_pair = fault_sets_from_faulty_agents(gold_faulty_agents)
        pred_detect, pred_type, pred_pair = fault_sets_from_faulty_agents(pred_faulty_agents)
        detect_gold.append(gold_detect)
        detect_pred.append(pred_detect)
        type_gold.append(gold_type)
        type_pred.append(pred_type)
        pair_gold.append(gold_pair)
        pair_pred.append(pred_pair)

    return {
        "detect": set_f1_metrics(detect_gold, detect_pred),
        "type": set_f1_metrics(type_gold, type_pred),
        "pair": set_f1_metrics(pair_gold, pair_pred),
    }
