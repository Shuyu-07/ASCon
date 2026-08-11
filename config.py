import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_DATA_ROOT = Path(os.environ.get("TRACE_DATA_ROOT", ROOT / "data"))

MODEL_CONFIG = {
    "step_dim": 256,
    "step_gru_hidden_dim": 128,
    "step_gru_layers": 1,
    "agent_dim": 128,
    "agent_hidden_dim": 256,
    "agent_query_dim": 128,
    "dgat_hidden_dim": 256,
    "dgat_layers": 2,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "epochs": 12,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "lambda_agent": 1.0,
    "lambda_step": 0.3,
    "validation_ratio": 0.1,
    "seed": 42,
}

TASK1_DATASET = {
    "name": "TracerTraj",
    "train": [ROOT / "AgentStepNet/AgentTracer/train.pt"],
    "test": [
        ROOT / "AgentStepNet/AgentTracer/code_test.pt",
        ROOT / "AgentStepNet/AgentTracer/math_test.pt",
        ROOT / "AgentStepNet/AgentTracer/agentic_test.pt",
    ],
    "subtests": {
        "code_test": ROOT / "AgentStepNet/AgentTracer/code_test.pt",
        "math_test": ROOT / "AgentStepNet/AgentTracer/math_test.pt",
        "agentic_test": ROOT / "AgentStepNet/AgentTracer/agentic_test.pt",
    },
}

TASK2_DATASETS = {
    "train": TRACE_DATA_ROOT / "Aegis" / "preprocessed" / "train_with_agent_error_labels.pt",
    "valid": TRACE_DATA_ROOT / "Aegis" / "preprocessed" / "val_with_agent_error_labels.pt",
    "test": TRACE_DATA_ROOT / "Aegis" / "preprocessed" / "test_with_agent_error_labels.pt",
    "wwtest": TRACE_DATA_ROOT / "Aegis" / "preprocessed" / "WWtest_with_agent_error_labels.pt",
    "label_map": TRACE_DATA_ROOT / "Aegis" / "preprocessed" / "agent_error_type_to_index.json",
    "display_name": "Aegis-Bench",
}
