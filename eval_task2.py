import argparse
import json
from pathlib import Path

import torch

from ASCon.config import TASK2_DATASETS
from ASCon.evaluate import add_binary_labels, evaluate_task2
from ASCon.models import Task2ASCon
from ASCon.utils import load_pt, write_json


def resolve_task2_datasets(data_dir=None):
    if data_dir is None:
        return TASK2_DATASETS
    data_dir = Path(data_dir)
    return {
        "valid": data_dir / "val_with_agent_error_labels.pt",
        "test": data_dir / "test_with_agent_error_labels.pt",
        "wwtest": data_dir / "WWtest_with_agent_error_labels.pt",
        "display_name": "Aegis-Bench",
    }


def evaluate_checkpoint(checkpoint_path, split, data_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = resolve_task2_datasets(data_dir)
    samples = add_binary_labels(load_pt(datasets[split]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Task2ASCon(
        checkpoint["step_text_dim"],
        checkpoint["role_dim"],
        checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_task2(model, samples)
    result = {
        "paper_model": "ASCon",
        "task": "task2",
        "dataset": datasets["display_name"],
        "split": split,
        "checkpoint": str(checkpoint_path),
        "faulty_agent": metrics["faulty_agent"],
        "error_mode": metrics["error_mode"],
        "agent_error_pair": metrics["agent_error_pair"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result, metrics["records"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ASCon Task 2 checkpoint.")
    parser.add_argument("--split", choices=["valid", "test", "wwtest"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    args = parser.parse_args()
    result, records = evaluate_checkpoint(args.checkpoint, args.split, data_dir=args.data_dir)
    if args.output_json is not None:
        write_json(args.output_json, {"result": result, "records": records})


if __name__ == "__main__":
    main()
