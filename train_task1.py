import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ASCon.config import TASK1_DATASET, TRAIN_CONFIG
from ASCon.evaluate import evaluate_task1
from ASCon.models import Task1ASCon
from ASCon.utils import REPORT_METRICS, load_samples, positive_weight, set_seed, split_data, write_json


def train(seed, epochs, output_dir):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples(TASK1_DATASET["train"])
    train_data, validation_data = split_data(samples, seed)
    test_data = load_samples(TASK1_DATASET["test"])
    subtest_data = {
        name: load_samples([path])
        for name, path in TASK1_DATASET["subtests"].items()
    }
    model = Task1ASCon(samples[0]["step_emb"].size(-1), samples[0]["agent_emb"].size(-1)).to(device)
    agent_loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight(train_data, "agent_y", device))
    step_loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight(train_data, "step_y", device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    destination = Path(output_dir) / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    best_path = destination / "best.pt"
    best_selection_score = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        started = time.perf_counter()
        losses = []
        for sample in train_data:
            step_y = sample["step_y"].float().to(device)
            agent_y = sample["agent_y"].float().to(device)
            output = model(sample)
            loss = (
                TRAIN_CONFIG["lambda_agent"] * agent_loss_fn(output["agent_logits"], agent_y)
                + TRAIN_CONFIG["lambda_step"] * step_loss_fn(output["step_logits"], step_y)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))

        epoch_seconds = time.perf_counter() - started
        validation_metrics, _ = evaluate_task1(model, validation_data, "validation")
        test_metrics, _ = evaluate_task1(model, test_data, "TracerTraj")
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "epoch_seconds": epoch_seconds,
            **{f"validation_{key}": validation_metrics[key] for key in REPORT_METRICS},
            **{f"test_{key}": test_metrics[key] for key in REPORT_METRICS},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        selection_score = validation_metrics["step_micro_accuracy"]
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset": "TracerTraj",
                    "seed": seed,
                    "epoch": epoch,
                    "selection_metric": "step_micro_accuracy",
                    "selection_score": best_selection_score,
                    "step_text_dim": samples[0]["step_emb"].size(-1),
                    "role_dim": samples[0]["agent_emb"].size(-1),
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation, _ = evaluate_task1(model, validation_data, "validation")
    final_test, records = evaluate_task1(model, test_data, "TracerTraj", save_predictions=True)
    subtest_metrics = {
        name: evaluate_task1(model, samples, name, save_predictions=False)[0]
        for name, samples in subtest_data.items()
    }
    total_samples = sum(metrics["num_samples"] for metrics in subtest_metrics.values())
    step_macro_accuracy = (
        sum(metrics["step_micro_accuracy"] for metrics in subtest_metrics.values()) / len(subtest_metrics)
        if subtest_metrics else 0.0
    )
    agent_macro_accuracy = (
        sum(metrics["agent_micro_accuracy"] for metrics in subtest_metrics.values()) / len(subtest_metrics)
        if subtest_metrics else 0.0
    )
    best_test = {
        "overall_accuracy": {
            "step_accuracy": (
                sum(metrics["step_micro_accuracy"] * metrics["num_samples"] for metrics in subtest_metrics.values()) / total_samples
                if total_samples else 0.0
            ),
            "agent_accuracy": (
                sum(metrics["agent_micro_accuracy"] * metrics["num_samples"] for metrics in subtest_metrics.values()) / total_samples
                if total_samples else 0.0
            ),
            "step_macro_accuracy": step_macro_accuracy,
            "agent_macro_accuracy": agent_macro_accuracy,
        },
        "subtests": {
            name: {
                "step_accuracy": metrics["step_micro_accuracy"],
                "agent_accuracy": metrics["agent_micro_accuracy"],
            }
            for name, metrics in subtest_metrics.items()
        },
    }
    result = {
        "paper_model": "ASCon",
        "task": "task1",
        "model": model.__class__.__name__,
        "dataset": "TracerTraj",
        "seed": seed,
        "epochs": epochs,
        "best_epoch": checkpoint["epoch"],
        "selection_metric": "step_micro_accuracy",
        "selection_score": checkpoint["selection_score"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "best_validation": final_validation,
        "best_test": best_test,
        "history": history,
    }
    write_json(destination / "result.json", result)
    with (destination / "epoch_metrics.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(destination / "test_predictions.json", records)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Train the ASCon Task 1 model.")
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--output_dir", default="ASCon/results/task1")
    args = parser.parse_args()
    train(
        seed=args.seed,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
