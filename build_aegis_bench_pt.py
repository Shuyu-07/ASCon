from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "Aegis-Bench"
DEFAULT_MODEL_NAME = "./Qwen3-Embedding-0.6B"
STEP_MAX_TOKENS = 1024
DEFAULT_INPUTS = {
    "train": DATA_DIR / "train_with_agent_error_labels.json",
    "valid": DATA_DIR / "val_with_agent_error_labels.json",
    "test": DATA_DIR / "test_with_agent_error_labels.json",
    "wwtest": DATA_DIR / "WWtest_with_agent_error_labels.json",
}
DEFAULT_OUTPUTS = {
    "train": DATA_DIR / "train_with_agent_error_labels.pt",
    "valid": DATA_DIR / "val_with_agent_error_labels.pt",
    "test": DATA_DIR / "test_with_agent_error_labels.pt",
    "wwtest": DATA_DIR / "WWtest_with_agent_error_labels.pt",
}
DEFAULT_LABEL_MAP_PATH = DATA_DIR / "agent_error_type_to_index.json"


class Qwen3Embedding:
    def __init__(
        self,
        model_name_or_path: str,
        use_cuda: bool = True,
        max_length: int = 8192,
        batch_size: int = 4,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.model = self._load_model(use_cuda=self.use_cuda)

    def _load_model(self, use_cuda: bool):
        model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
        )
        if use_cuda:
            model = model.cuda()
        model.eval()
        return model

    def _switch_to_cpu(self) -> None:
        self.use_cuda = False
        try:
            del self.model
        except AttributeError:
            pass
        self.model = self._load_model(use_cuda=False)

    def _last_token_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def encode(
        self,
        sentences: list[str] | str,
        dim: int = -1,
    ) -> Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        outputs: list[Tensor] = []
        start = 0
        while start < len(sentences):
            batch = sentences[start:start + self.batch_size]
            try:
                batch_output = self._encode_once(batch, dim=dim)
            except Exception as exc:
                message = str(exc).lower()
                is_cuda_error = any(
                    keyword in message
                    for keyword in ("cuda", "cublas", "acceleratorerror", "illegal memory access")
                )
                if not is_cuda_error:
                    raise
                if self.use_cuda:
                    self._switch_to_cpu()
                    batch_output = self._encode_once(batch, dim=dim)
                else:
                    raise
            outputs.append(batch_output.cpu())
            start += self.batch_size
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 0), dtype=torch.float32)

    def _encode_once(self, sentences: list[str], dim: int = -1) -> Tensor:
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            model_outputs = self.model(**inputs)
            output = self._last_token_pool(
                model_outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            if dim > 0:
                output = output[:, :dim]
        return output

def read_json(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array")
    return rows


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def truncate_by_tokens(text: str, tokenizer, max_tokens: int = STEP_MAX_TOKENS) -> str:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_agent_edges(step_agent_ids: list[int]) -> torch.Tensor:
    edges = [(step_agent_ids[i], step_agent_ids[i + 1]) for i in range(len(step_agent_ids) - 1)]
    edges = sorted(set(edges))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def select_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    history = list(row.get("history", []))
    history.sort(key=lambda step: int(step.get("step_id", step.get("step", 0))))
    return history


def infer_label_mapping(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    label_to_index: dict[str, int] = {}
    for rows in rows_by_split.values():
        for row in rows:
            names = row.get("gold_agent_error_type_names", [])
            ids = row.get("gold_agent_error_type_ids", [])
            if len(names) != len(ids):
                raise ValueError("gold_agent_error_type_names and ids length mismatch")
            for name, index in zip(names, ids):
                normalized_name = normalize_text(name)
                index = int(index)
                if normalized_name in label_to_index and label_to_index[normalized_name] != index:
                    raise ValueError(
                        f"inconsistent label id for {normalized_name}: "
                        f"{label_to_index[normalized_name]} vs {index}"
                    )
                label_to_index[normalized_name] = index
    if "benign" not in label_to_index:
        raise ValueError("label mapping must contain benign")
    return dict(sorted(label_to_index.items(), key=lambda item: item[1]))


def build_sample(
    row: dict[str, Any],
    encoder: Qwen3Embedding,
    num_classes: int,
) -> dict[str, Any]:
    history = select_history(row)
    step_ids: list[int] = []
    step_contents: list[str] = []
    step_agent_names: list[str] = []
    step_labels: list[int] = []

    for index, step in enumerate(history):
        step_id = int(step.get("step_id", step.get("step", index)))
        agent_name = normalize_text(step.get("name")) or "unknown_agent"
        step_ids.append(step_id)
        step_agent_names.append(agent_name)
        step_contents.append(
            truncate_by_tokens(
                normalize_text(step.get("content")),
                tokenizer=encoder.tokenizer,
            )
        )
        step_labels.append(int(step.get("is_mistake", 0)))

    agent_names: list[str] = []
    for agent_name in step_agent_names:
        if agent_name not in agent_names:
            agent_names.append(agent_name)
    agent_to_index = {name: index for index, name in enumerate(agent_names)}
    step_agent_ids = [agent_to_index[name] for name in step_agent_names]

    gold_agent_error_type_names = [normalize_text(name) or "benign" for name in row.get("gold_agent_error_type_names", [])]
    gold_agent_error_type_ids = [int(index) for index in row.get("gold_agent_error_type_ids", [])]
    if len(gold_agent_error_type_names) != len(agent_names) or len(gold_agent_error_type_ids) != len(agent_names):
        raise ValueError(
            f"agent label length mismatch for {row.get('file')}: "
            f"{len(agent_names)} agents, "
            f"{len(gold_agent_error_type_names)} names, "
            f"{len(gold_agent_error_type_ids)} ids"
        )

    step_emb = encoder.encode(step_contents).float().cpu()
    agent_emb = encoder.encode(agent_names).float().cpu()
    agent_y = [1 if label_id > 0 else 0 for label_id in gold_agent_error_type_ids]

    return {
        "file": normalize_text(row.get("file")),
        "question": normalize_text(row.get("question")),
        "ground_truth": row.get("ground_truth"),
        "mistake_step": [],
        "mistake_agent": row.get("mistake_agent", []),
        "step_ids": torch.tensor(step_ids, dtype=torch.long),
        "step_contents": step_contents,
        "step_emb": step_emb,
        "step_y": torch.tensor(step_labels, dtype=torch.float),
        "agent_names": agent_names,
        "agent_emb": agent_emb,
        "agent_y": torch.tensor(agent_y, dtype=torch.float),
        "step_agent_ids": torch.tensor(step_agent_ids, dtype=torch.long),
        "agent_edge_index": build_agent_edges(step_agent_ids),
        "agent_error_type_names": gold_agent_error_type_names,
        "agent_error_type_ids": torch.tensor(gold_agent_error_type_ids, dtype=torch.long),
        "agent_error_type_num_classes": num_classes,
    }


def build_split(
    split: str,
    input_path: Path,
    output_path: Path,
    encoder: Qwen3Embedding,
    num_classes: int,
) -> dict[str, Any]:
    rows = read_json(input_path)
    samples = [build_sample(row, encoder, num_classes) for row in tqdm(rows, desc=f"build_{split}")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_path)
    return {
        "input": str(input_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "num_samples": len(samples),
    }


def save_label_mapping(path: Path, label_to_index: dict[str, int]) -> None:
    payload = {
        "num_classes": len(label_to_index),
        "label_to_index": label_to_index,
        "index_to_label": {str(index): label for label, index in label_to_index.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Aegis-Bench PT files for ASCon Task 2.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(DEFAULT_INPUTS),
        default=list(DEFAULT_INPUTS),
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    rows_by_split = {split: read_json(DEFAULT_INPUTS[split]) for split in DEFAULT_INPUTS}
    label_to_index = infer_label_mapping(rows_by_split)
    save_label_mapping(DEFAULT_LABEL_MAP_PATH, label_to_index)

    use_cuda = torch.cuda.is_available()
    if args.device == "cpu":
        use_cuda = False
    elif args.device == "cuda":
        use_cuda = True

    encoder = Qwen3Embedding(
        model_name_or_path=args.model_name,
        use_cuda=use_cuda,
        max_length=STEP_MAX_TOKENS,
        batch_size=args.batch_size,
    )
    report = {
        "label_map": str(DEFAULT_LABEL_MAP_PATH.relative_to(ROOT)),
        "num_classes": len(label_to_index),
    }
    for split in args.splits:
        report[split] = build_split(
            split=split,
            input_path=DEFAULT_INPUTS[split],
            output_path=DEFAULT_OUTPUTS[split],
            encoder=encoder,
            num_classes=len(label_to_index),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
