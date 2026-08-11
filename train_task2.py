import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ASCon.config import TASK2_DATASETS, TRAIN_CONFIG
from ASCon.evaluate import add_binary_labels, evaluate_task2
from ASCon.models import Task2ASCon
from ASCon.utils import load_pt, positive_weight, set_seed, write_json


def resolve_task2_datasets(data_dir=None):
    if data_dir is None:
        return TASK2_DATASETS
    data_dir = Path(data_dir)
    return {
        "train": data_dir / "train_with_agent_error_labels.pt",
        "valid": data_dir / "val_with_agent_error_labels.pt",
        "test": data_dir / "test_with_agent_error_labels.pt",
        "display_name": "Aegis-Bench",
    }


def train(seed, epochs, output_dir, data_dir=None):
    set_seed(seed)
    datasets = resolve_task2_datasets(data_dir)
    train_samples = add_binary_labels(load_pt(datasets["train"]))
    valid_samples = add_binary_labels(load_pt(datasets["valid"]))
    test_samples = add_binary_labels(load_pt(datasets["test"]))
    num_classes = int(train_samples[0]["agent_error_type_num_classes"])
    step_text_dim = train_samples[0]["step_emb"].size(-1)
    role_dim = train_samples[0]["agent_emb"].size(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Task2ASCon(step_text_dim, role_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    binary_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight(train_samples, "agent_error_binary", device)
    )
    multiclass_loss_fn = nn.CrossEntropyLoss()
    destination = Path(output_dir) / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    best_valid_pair_macro_f1 = -1.0
    best_path = destination / "best.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_samples)
        started = time.perf_counter()
        losses = []
        for sample in train_samples:
            output = model(sample)
            gold_type = sample["agent_error_type_ids"].long().to(device)
            gold_binary = sample["agent_error_binary"].float().to(device)
            binary_loss = binary_loss_fn(output["binary_logits"], gold_binary)
            faulty_mask = gold_type > 0
            if faulty_mask.any():
                type_loss = multiclass_loss_fn(
                    output["fault_type_logits"][faulty_mask],
                    gold_type[faulty_mask] - 1,
                )
            else:
                type_loss = torch.tensor(0.0, device=device)
            loss = binary_loss + type_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))

        epoch_seconds = time.perf_counter() - started
        valid_metrics = evaluate_task2(model, valid_samples)
        test_metrics = evaluate_task2(model, test_samples)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "epoch_seconds": epoch_seconds,
            "valid_faulty_agent_micro_f1": valid_metrics["faulty_agent"]["micro_f1"],
            "valid_faulty_agent_macro_f1": valid_metrics["faulty_agent"]["macro_f1"],
            "valid_error_mode_micro_f1": valid_metrics["error_mode"]["micro_f1"],
            "valid_error_mode_macro_f1": valid_metrics["error_mode"]["macro_f1"],
            "valid_agent_error_pair_micro_f1": valid_metrics["agent_error_pair"]["micro_f1"],
            "valid_agent_error_pair_macro_f1": valid_metrics["agent_error_pair"]["macro_f1"],
            "test_faulty_agent_micro_f1": test_metrics["faulty_agent"]["micro_f1"],
            "test_faulty_agent_macro_f1": test_metrics["faulty_agent"]["macro_f1"],
            "test_error_mode_micro_f1": test_metrics["error_mode"]["micro_f1"],
            "test_error_mode_macro_f1": test_metrics["error_mode"]["macro_f1"],
            "test_agent_error_pair_micro_f1": test_metrics["agent_error_pair"]["micro_f1"],
            "test_agent_error_pair_macro_f1": test_metrics["agent_error_pair"]["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if valid_metrics["agent_error_pair"]["macro_f1"] > best_valid_pair_macro_f1:
            best_valid_pair_macro_f1 = valid_metrics["agent_error_pair"]["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_valid_pair_macro_f1": best_valid_pair_macro_f1,
                    "step_text_dim": step_text_dim,
                    "role_dim": role_dim,
                    "num_classes": num_classes,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_valid = evaluate_task2(model, valid_samples)
    final_test = evaluate_task2(model, test_samples)
    result = {
        "paper_model": "ASCon",
        "task": "task2",
        "model": model.__class__.__name__,
        "dataset": datasets["display_name"],
        "epochs": epochs,
        "best_epoch": checkpoint["epoch"],
        "best_valid_pair_macro_f1": checkpoint["best_valid_pair_macro_f1"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "best_validation": {
            "faulty_agent": final_valid["faulty_agent"],
            "error_mode": final_valid["error_mode"],
            "agent_error_pair": final_valid["agent_error_pair"],
        },
        "best_test": {
            "faulty_agent": final_test["faulty_agent"],
            "error_mode": final_test["error_mode"],
            "agent_error_pair": final_test["agent_error_pair"],
        },
        "history": history,
    }
    write_json(destination / "result.json", result)
    write_json(destination / "test_predictions.json", final_test["records"])
    with (destination / "epoch_metrics.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Train the ASCon Task 2 model.")
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--output_dir", default="ASCon/results/task2")
    parser.add_argument("--data_dir", type=Path, default=None)
    args = parser.parse_args()
    train(seed=args.seed, epochs=args.epochs, output_dir=args.output_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
