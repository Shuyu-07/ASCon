import argparse
import json
from pathlib import Path

import torch

from ASCon.config import TASK1_DATASET
from ASCon.evaluate import evaluate_task1
from ASCon.models import Task1ASCon
from ASCon.utils import load_samples, write_json


def evaluate_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = load_samples(TASK1_DATASET["test"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Task1ASCon(checkpoint["step_text_dim"], checkpoint["role_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, records = evaluate_task1(model, test_data, "TracerTraj", save_predictions=True)
    subtest_data = {
        name: load_samples([path])
        for name, path in TASK1_DATASET["subtests"].items()
    }
    subtest_metrics = {
        name: evaluate_task1(model, samples, name, save_predictions=False)[0]
        for name, samples in subtest_data.items()
    }
    total_samples = sum(item["num_samples"] for item in subtest_metrics.values())
    step_macro_accuracy = (
        sum(item["step_micro_accuracy"] for item in subtest_metrics.values()) / len(subtest_metrics)
        if subtest_metrics else 0.0
    )
    agent_macro_accuracy = (
        sum(item["agent_micro_accuracy"] for item in subtest_metrics.values()) / len(subtest_metrics)
        if subtest_metrics else 0.0
    )
    result = {
        "paper_model": "ASCon",
        "task": "task1",
        "dataset": "TracerTraj",
        "checkpoint": str(checkpoint_path),
        "metrics": {
            "overall_accuracy": {
                "step_accuracy": (
                    sum(item["step_micro_accuracy"] * item["num_samples"] for item in subtest_metrics.values()) / total_samples
                    if total_samples else 0.0
                ),
                "agent_accuracy": (
                    sum(item["agent_micro_accuracy"] * item["num_samples"] for item in subtest_metrics.values()) / total_samples
                    if total_samples else 0.0
                ),
                "step_macro_accuracy": step_macro_accuracy,
                "agent_macro_accuracy": agent_macro_accuracy,
            },
            "subtests": {
                name: {
                    "step_accuracy": item["step_micro_accuracy"],
                    "agent_accuracy": item["agent_micro_accuracy"],
                }
                for name, item in subtest_metrics.items()
            },
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, records


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ASCon Task 1 checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()
    result, records = evaluate_checkpoint(args.checkpoint)
    if args.output_json is not None:
        write_json(args.output_json, {"result": result, "records": records})


if __name__ == "__main__":
    main()
