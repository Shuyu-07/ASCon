from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import torch
from openai import OpenAI

from ASCon.evaluate import AEGIS_ERROR_TYPES
from ASCon.metrics import compute_aegis_fault_metrics_from_faulty_agents_pairs
from ASCon.models import Task2ASCon
from ASCon.prompt_template import (
    ASCON_ENHANCED_AEGIS_COT_PROMPT,
    build_chat_content,
    build_chat_content_with_probability,
    build_reference_content,
    build_task1_prompt,
)
from ASCon.utils import load_pt

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = ROOT / "Aegis-Bench" / "WWtest_with_agent_error_labels.json"
DEFAULT_INPUT_PT = ROOT / "Aegis-Bench" / "WWtest_with_agent_error_labels.pt"
DEFAULT_CHECKPOINT = ROOT / "results" / "task2_aegis_bench_local" / "seed_42" / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task2_aegis_enhanced_llm"
DEFAULT_AUX_OUTPUT_JSON = ROOT / "FaultProbabilityFile" / "Task2-WWtest-Agent-fault-probs.json"
DEFAULT_TASK1_OUTPUT_DIR = ROOT / "results" / "task1_root_enhanced_llm"
DEFAULT_TASK1_ALGORITHM_DIR = ROOT / "WhoWhen-RootTest" / "Algorithm-Generated"
DEFAULT_TASK1_HANDCRAFT_DIR = ROOT / "WhoWhen-RootTest" / "Hand-Craft"
DEFAULT_TASK1_ALGORITHM_PROBS = ROOT / "FaultProbabilityFile" / "Task1-WWtest-Algorithm_ranked_probs.json"
DEFAULT_TASK1_HANDCRAFT_PROBS = ROOT / "FaultProbabilityFile" / "Task1-WWtest-Handcraft_ranked_probs.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_label_map(path: Path) -> Dict[int, str]:
    payload = load_json(path)
    index_to_label = payload.get("index_to_label", {})
    return {int(index): str(label) for index, label in index_to_label.items()}


def normalize_error_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"fm\s*-\s*(\d+)\s*\.\s*(\d+)", text, re.I)
    if match:
        return f"FM-{match.group(1)}.{match.group(2)}"
    return re.sub(r"\s+", "", text).upper()


def normalize_agent_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip()
    return re.sub(r"\s+", " ", name).casefold()


def normalize_faulty_agents(items: Any) -> List[Dict[str, str]]:
    if not isinstance(items, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        agent_name = str(item.get("agent_name", "")).strip()
        error_type = normalize_error_type(item.get("error_type", ""))
        if agent_name and error_type:
            normalized.append({"agent_name": agent_name, "error_type": error_type})
    return normalized


def find_faulty_agents_payload(payload: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(payload, dict):
        if "faulty_agents" in payload:
            result = normalize_faulty_agents(payload["faulty_agents"])
            if result or payload["faulty_agents"] == []:
                return result
        for value in payload.values():
            nested = find_faulty_agents_payload(value)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        result = normalize_faulty_agents(payload)
        if result or payload == []:
            return result
        for item in payload:
            nested = find_faulty_agents_payload(item)
            if nested is not None:
                return nested
    return None


def parse_faulty_agents_from_response(text: str) -> List[Dict[str, str]]:
    if not text.strip():
        return []
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    candidates.extend(fenced)
    candidates.append(text)
    json_like = re.findall(r"(\{[\s\S]*\})", text)
    candidates.extend(json_like)
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        result = find_faulty_agents_payload(payload)
        if result is not None:
            return result
    return []


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {"", "null", "none"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_root_fault_response(raw_response: str) -> Dict[str, Any]:
    if not raw_response:
        return {}

    payload: Dict[str, Any] = {}
    patterns = {
        "fault_agent": [
            r"(?im)^\s*Agent role\s*:\s*(.+?)\s*$",
            r"(?im)^\s*Agent\s*:\s*(.+?)\s*$",
            r"(?im)\"agent_name\"\s*:\s*\"([^\"]+)\"",
        ],
        "fault_step": [
            r"(?im)^\s*Step Number\s*:\s*(.+?)\s*$",
            r"(?im)^\s*Step\s*:\s*(.+?)\s*$",
            r"(?im)\"step_number\"\s*:\s*([0-9]+|null|none)",
        ],
        "reason": [
            r"(?ims)^\s*Reason for Mistake\s*:\s*(.+)\s*$",
            r"(?ims)^\s*Reason\s*:\s*(.+)\s*$",
            r"(?ims)\"reason_for_mistake\"\s*:\s*\"([^\"]*)\"",
        ],
    }
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, raw_response)
            if match:
                payload[key] = match.group(1).strip()
                break
    step_value = parse_optional_int(payload.get("fault_step"))
    agent_value = payload.get("fault_agent")
    reason_value = payload.get("reason")
    return {
        "fault_agent": None if agent_value is None else str(agent_value).strip(),
        "fault_step": step_value,
        "reason": "" if reason_value is None else str(reason_value).strip(),
    }


def normalize_root_prediction(parsed: Dict[str, Any]) -> Dict[str, Any]:
    fault_agent = parsed.get("fault_agent")
    fault_step = parsed.get("fault_step")
    return {
        "fault_detected": fault_agent is not None or fault_step is not None,
        "fault_agent": None if fault_agent in {"", "none", "null"} else fault_agent,
        "fault_step": parse_optional_int(fault_step),
    }


def build_conversation_text(row: Dict[str, Any], auxiliary_agents: List[Dict[str, Any]]) -> str:
    lines = [f"QUERY:\n{str(row.get('question', '')).strip()}", "", "CONVERSATION HISTORY:"]
    history = sorted(row.get("history", []), key=lambda item: int(item.get("step_id", 0) or 0))
    for entry in history:
        step_id = entry.get("step_id", "")
        agent_name = str(entry.get("name", "")).strip()
        role = str(entry.get("role", "")).strip()
        content = str(entry.get("content", "")).strip()
        header = f"Step {step_id} - {agent_name}"
        if role:
            header += f" [{role}]"
        lines.append(f"{header}:\n{content}")
        lines.append("")
    for agent in auxiliary_agents:
        top_5 = "、".join(item["error_type"] for item in agent.get("top_5_likely_fault_types", []))
        lines.append("{")
        lines.append(f"\"agent_name\": \"{agent['agent_name']}\",")
        lines.append(f"\"fault_probability\": {agent['fault_probability']},")
        lines.append(f"\"top_5_likely_fault_types\": {top_5}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).strip()


def build_auxiliary_profiles(
    checkpoint_path: Path,
    pt_path: Path,
    label_map_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    samples = load_pt(pt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Task2ASCon(
        checkpoint["step_text_dim"],
        checkpoint["role_dim"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    index_to_label = load_label_map(label_map_path)
    profiles: Dict[str, List[Dict[str, Any]]] = {}
    with torch.no_grad():
        for sample in samples:
            output = model(sample)
            binary_prob = torch.sigmoid(output["binary_logits"]).cpu().tolist()
            type_prob = torch.softmax(output["fault_type_logits"], dim=-1).cpu()
            agents: List[Dict[str, Any]] = []
            for agent_name, fault_prob, type_vector in zip(sample["agent_names"], binary_prob, type_prob):
                top_values, top_indices = torch.topk(type_vector, k=min(5, type_vector.numel()))
                top_types = []
                for score, index in zip(top_values.tolist(), top_indices.tolist()):
                    label_id = int(index) + 1
                    top_types.append(
                        {
                            "error_type": index_to_label.get(label_id, AEGIS_ERROR_TYPES[label_id]),
                            "probability": float(score),
                        }
                    )
                agents.append(
                    {
                        "agent_name": str(agent_name),
                        "fault_probability": float(fault_prob),
                        "top_5_likely_fault_types": top_types,
                    }
                )
            profiles[Path(str(sample["file"])).name] = agents
    return profiles


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON array")
    for index, row in enumerate(rows):
        row["__file_name__"] = Path(str(row.get("file", f"{index:04d}.json"))).name
        row["__index__"] = index
        row["__aux_key__"] = str(row.get("question_ID") or row.get("file") or row["__file_name__"]).strip()
    return rows


def export_auxiliary_profiles(
    rows: List[Dict[str, Any]],
    auxiliary_profiles: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for row in rows:
        file_name = row["__file_name__"]
        agents = auxiliary_profiles.get(file_name, [])
        payload.append(
            {
                "file": row["__aux_key__"],
                "agents": agents,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def auxiliary_lookup_from_exported_payload(
    exported_payload: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_key = {
        str(item.get("file", "")).strip(): item.get("agents", [])
        for item in exported_payload
        if isinstance(item, dict)
    }
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        agents = by_key.get(row["__aux_key__"], [])
        lookup[row["__aux_key__"]] = agents
        lookup[row["__file_name__"]] = agents
    return lookup


@dataclass
class ModelSpec:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    disable_thinking: bool = False
    max_retries: int = 3
    min_interval_seconds: float = 0.0


@dataclass
class Runner:
    client: Optional[OpenAI]
    provider: str
    model: str
    api_key: str
    max_tokens: int = 4096
    max_retries: int = 3
    min_interval_seconds: float = 0.0
    last_request_at: float = 0.0


def build_model_spec(model_name: str) -> ModelSpec:
    model_name = model_name.strip()
    if model_name == "gpt-4o-mini":
        return ModelSpec("openai", model_name, os.environ["OPENAI_API_KEY"])
    if model_name == "deepseek-v4-pro":
        return ModelSpec(
            "deepseek",
            model_name,
            os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            disable_thinking=True,
        )
    if model_name.startswith("gemini"):
        return ModelSpec("gemini", model_name, os.environ["GEMINI_API_KEY"])
    if model_name.startswith("qwen"):
        return ModelSpec(
            "qwen",
            model_name,
            os.environ["QWEN_API_KEY"],
            base_url=os.environ.get("QWEN_OPENAI_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def make_runner(spec: ModelSpec) -> Runner:
    if spec.provider == "gemini":
        return Runner(
            client=None,
            provider=spec.provider,
            model=spec.model,
            api_key=spec.api_key,
            max_retries=spec.max_retries,
            min_interval_seconds=spec.min_interval_seconds,
        )
    kwargs: Dict[str, Any] = {"api_key": spec.api_key}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    return Runner(
        client=OpenAI(**kwargs),
        provider=spec.provider,
        model=spec.model,
        api_key=spec.api_key,
        max_retries=spec.max_retries,
        min_interval_seconds=spec.min_interval_seconds,
    )


def make_runner_factory(spec: ModelSpec):
    local_state = threading.local()

    def get_runner() -> Runner:
        runner = getattr(local_state, "runner", None)
        if runner is None:
            runner = make_runner(spec)
            local_state.runner = runner
        return runner

    return get_runner


def call_llm(runner: Runner, prompt: str, temperature: float = 0.0) -> str:
    if runner.provider == "gemini":
        last_error: Optional[Exception] = None
        for attempt in range(runner.max_retries):
            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{runner.model}:generateContent?key={runner.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": temperature},
                    },
                    timeout=180,
                )
                response.raise_for_status()
                payload = response.json()
                texts: List[str] = []
                for candidate in payload.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            texts.append(part["text"])
                return "\n".join(texts).strip()
            except Exception as exc:
                last_error = exc
                if attempt == runner.max_retries - 1:
                    break
                time.sleep(min(60.0, 2.0 * (attempt + 1)))
        raise last_error if last_error is not None else RuntimeError("Gemini request failed")

    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in MAS failure attribution."},
        {"role": "user", "content": prompt},
    ]
    kwargs: Dict[str, Any] = {
        "model": runner.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": runner.max_tokens,
    }
    if runner.provider == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    response = runner.client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def evaluate_records(
    rows: List[Dict[str, Any]],
    auxiliary_profiles: Dict[str, List[Dict[str, Any]]],
    spec: ModelSpec,
    output_dir: Path,
    workers: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    runner_getter = make_runner_factory(spec)
    write_lock = threading.Lock()
    existing: Dict[str, Dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["file"]] = record

    def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        file_name = row["__file_name__"]
        if file_name in existing:
            return None
        auxiliary_agents = auxiliary_profiles.get(row["__aux_key__"], auxiliary_profiles.get(file_name, []))
        conversation_text = build_conversation_text(row, auxiliary_agents)
        prompt = ASCON_ENHANCED_AEGIS_COT_PROMPT.format(conversation_text=conversation_text)
        raw_response = call_llm(runner_getter(), prompt=prompt, temperature=0.0)
        pred_faulty_agents = parse_faulty_agents_from_response(raw_response)
        gold_faulty_agents = normalize_faulty_agents(
            [
                {"agent_name": agent_name, "error_type": error_type}
                for agent_name, error_type in zip(
                    row.get("mistake_agent", []),
                    row.get("gold_agent_error_type_names", []),
                )
                if normalize_error_type(error_type) and normalize_error_type(error_type) != "BENIGN"
            ]
        )
        return {
            "file": file_name,
            "index": row["__index__"],
            "question": row.get("question", ""),
            "conversation_text": conversation_text,
            "gold_faulty_agents": gold_faulty_agents,
            "pred_faulty_agents": pred_faulty_agents,
            "raw_response": raw_response,
        }

    pending = [row for row in rows if row["__file_name__"] not in existing]
    if workers <= 1:
        for row in pending:
            record = process_row(row)
            if record is None:
                continue
            existing[record["file"]] = record
            with records_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(process_row, row): row for row in pending}
            for future in as_completed(future_map):
                record = future.result()
                if record is None:
                    continue
                with write_lock:
                    existing[record["file"]] = record
                    with records_path.open("a", encoding="utf-8") as sink:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = sorted(existing.values(), key=lambda item: item["index"])
    gold = [record["gold_faulty_agents"] for record in records]
    pred = [record["pred_faulty_agents"] for record in records]
    metrics = compute_aegis_fault_metrics_from_faulty_agents_pairs(gold, pred)
    summary = {
        "model": spec.model,
        "samples": len(records),
        "faulty_agent": metrics["detect"],
        "error_mode": metrics["type"],
        "agent_error_pair": metrics["pair"],
        "records_path": str(records_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_task1_directory_samples(dataset_dir: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    json_paths = sorted(dataset_dir.glob("*.json"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)
    for index, path in enumerate(json_paths):
        sample = load_json(path)
        sample["__file_name__"] = path.name
        sample["__line_no__"] = index + 1
        samples.append(sample)
    return samples


def load_task1_ranked_prob_records(algorithm_path: Path, handcrafted_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    records_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name, path in (
        ("algorithm", algorithm_path),
        ("handcrafted", handcrafted_path),
    ):
        payload = load_json(path)
        records_by_file: Dict[str, Dict[str, Any]] = {}
        for record in payload.get("records", []):
            file_name = Path(str(record.get("file", ""))).name
            if file_name:
                records_by_file[file_name] = record
        records_by_dataset[dataset_name] = records_by_file
    return records_by_dataset


def evaluate_task1_dataset(
    dataset_name: str,
    dataset_dir: Path,
    top_n: int,
    agent_top_n: int,
    include_agents: bool,
    prompt_kind: str,
    reference_format: str,
    ranked_records: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
    spec: ModelSpec,
    workers: int,
) -> Dict[str, Any]:
    samples = load_task1_directory_samples(dataset_dir)
    dataset_out_dir = output_dir / dataset_name / spec.provider / spec.model
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    records_path = dataset_out_dir / "records.jsonl"
    summary_path = dataset_out_dir / "summary.json"

    existing: Dict[str, Dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["record_key"]] = record

    runner_getter = make_runner_factory(spec)
    write_lock = threading.Lock()

    def process_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        file_name = sample["__file_name__"]
        if file_name in existing:
            return None
        ranked_record = ranked_records[dataset_name][file_name]
        chat_content = build_chat_content(sample.get("history", []))
        chat_content_with_probability = build_chat_content_with_probability(sample.get("history", []), ranked_record)
        reference_content = build_reference_content(
            ranked_record,
            top_n=top_n,
            agent_top_n=agent_top_n,
            include_agents=include_agents,
            reference_format=reference_format,
            history=sample.get("history", []),
        )
        problem = str(sample.get("question", "")).strip()

        raw_response = ""
        parse_error = None
        parsed: Dict[str, Any] = {}
        try:
            prompt = build_task1_prompt(
                problem=problem,
                chat_content=chat_content,
                reference_content=reference_content,
                prompt_kind=prompt_kind,
                chat_content_with_probability=chat_content_with_probability,
            )
            raw_response = call_llm(runner_getter(), prompt=prompt, temperature=0.0) or ""
            parsed = parse_root_fault_response(raw_response)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        normalized = normalize_root_prediction(parsed)
        return {
            "record_key": file_name,
            "line_no": sample["__line_no__"],
            "file": file_name,
            "dataset_name": dataset_name,
            "problem": problem,
            "reference_content": json.loads(reference_content),
            "chat_content": json.loads(chat_content),
            "chat_content_with_probability": json.loads(chat_content_with_probability),
            "raw_response": raw_response,
            "parsed_response": parsed if parsed else None,
            "normalized_prediction": normalized,
            "pred_fault_detected": normalized["fault_detected"],
            "pred_fault_step": normalized["fault_step"],
            "pred_fault_agent": normalized["fault_agent"],
            "gold_fault_step": parse_optional_int(sample.get("mistake_step")),
            "gold_fault_agent": None if sample.get("mistake_agent") is None else str(sample.get("mistake_agent")),
            "parse_error": parse_error,
        }

    pending_samples = [sample for sample in samples if sample["__file_name__"] not in existing]
    if workers <= 1:
        for sample in pending_samples:
            record = process_sample(sample)
            if record is None:
                continue
            existing[record["record_key"]] = record
            with records_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(process_sample, sample): sample for sample in pending_samples}
            for future in as_completed(future_map):
                record = future.result()
                if record is None:
                    continue
                with write_lock:
                    existing[record["record_key"]] = record
                    with records_path.open("a", encoding="utf-8") as sink:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    ordered = sorted(existing.values(), key=lambda item: item["line_no"])
    detected_count = sum(1 for item in ordered if item["pred_fault_detected"])
    parsed_count = sum(1 for item in ordered if item.get("parsed_response") is not None)
    step_correct_count = sum(
        1 for item in ordered
        if item.get("pred_fault_step") is not None
        and item.get("gold_fault_step") is not None
        and int(item["pred_fault_step"]) == int(item["gold_fault_step"])
    )
    agent_correct_count = sum(
        1 for item in ordered
        if item.get("pred_fault_agent") is not None
        and item.get("gold_fault_agent") is not None
        and str(item["pred_fault_agent"]).strip() == str(item["gold_fault_agent"]).strip()
    )
    joint_correct_count = sum(
        1 for item in ordered
        if item.get("pred_fault_step") is not None
        and item.get("gold_fault_step") is not None
        and item.get("pred_fault_agent") is not None
        and item.get("gold_fault_agent") is not None
        and int(item["pred_fault_step"]) == int(item["gold_fault_step"])
        and str(item["pred_fault_agent"]).strip() == str(item["gold_fault_agent"]).strip()
    )
    summary = {
        "dataset": dataset_name,
        "provider": spec.provider,
        "model": spec.model,
        "prompt_kind": prompt_kind,
        "top_n": top_n,
        "agent_top_n": agent_top_n,
        "samples": len(ordered),
        "pred_fault_detected_count": detected_count,
        "parsed_count": parsed_count,
        "parse_failures": sum(1 for item in ordered if item.get("parse_error")),
        "step_correct_count": step_correct_count,
        "step_accuracy": (step_correct_count / len(ordered)) if ordered else 0.0,
        "agent_correct_count": agent_correct_count,
        "agent_accuracy": (agent_correct_count / len(ordered)) if ordered else 0.0,
        "joint_correct_count": joint_correct_count,
        "joint_accuracy": (joint_correct_count / len(ordered)) if ordered else 0.0,
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_task1_combined_summary(dataset_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_samples = sum(item["samples"] for item in dataset_summaries)
    step_correct_total = sum(item["step_correct_count"] for item in dataset_summaries)
    agent_correct_total = sum(item["agent_correct_count"] for item in dataset_summaries)
    joint_correct_total = sum(item["joint_correct_count"] for item in dataset_summaries)
    return {
        "sample_total": total_samples,
        "step": {
            "correct_count": step_correct_total,
            "micro_accuracy": (step_correct_total / total_samples) if total_samples else 0.0,
            "macro_accuracy": (sum(item["step_accuracy"] for item in dataset_summaries) / len(dataset_summaries)) if dataset_summaries else 0.0,
        },
        "agent": {
            "correct_count": agent_correct_total,
            "micro_accuracy": (agent_correct_total / total_samples) if total_samples else 0.0,
            "macro_accuracy": (sum(item["agent_accuracy"] for item in dataset_summaries) / len(dataset_summaries)) if dataset_summaries else 0.0,
        },
        "joint": {
            "correct_count": joint_correct_total,
            "micro_accuracy": (joint_correct_total / total_samples) if total_samples else 0.0,
            "macro_accuracy": (sum(item["joint_accuracy"] for item in dataset_summaries) / len(dataset_summaries)) if dataset_summaries else 0.0,
        },
    }


def main_task2() -> None:
    parser = argparse.ArgumentParser(description="Run the ASCon-enhanced LLM evaluation for Aegis-Bench Task 2.")
    parser.add_argument("--input_json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--input_pt", type=Path, default=DEFAULT_INPUT_PT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--label_map", type=Path, default=ROOT / "Aegis-Bench" / "agent_error_type_to_index.json")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aux_output_json", type=Path, default=DEFAULT_AUX_OUTPUT_JSON)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.input_json)
    raw_auxiliary_profiles = build_auxiliary_profiles(
        checkpoint_path=args.checkpoint,
        pt_path=args.input_pt,
        label_map_path=args.label_map,
    )
    exported_payload = export_auxiliary_profiles(rows, raw_auxiliary_profiles, args.aux_output_json)
    auxiliary_profiles = auxiliary_lookup_from_exported_payload(exported_payload, rows)
    spec = build_model_spec(args.model)
    output_dir = args.output_dir / args.model
    summary = evaluate_records(rows, auxiliary_profiles, spec, output_dir, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main_task1() -> None:
    parser = argparse.ArgumentParser(description="Run the ASCon-enhanced LLM evaluation for WhoWhen Task 1.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TASK1_OUTPUT_DIR)
    parser.add_argument("--algorithm-dir", type=Path, default=DEFAULT_TASK1_ALGORITHM_DIR)
    parser.add_argument("--handcrafted-dir", type=Path, default=DEFAULT_TASK1_HANDCRAFT_DIR)
    parser.add_argument("--algorithm-probs", type=Path, default=DEFAULT_TASK1_ALGORITHM_PROBS)
    parser.add_argument("--handcrafted-probs", type=Path, default=DEFAULT_TASK1_HANDCRAFT_PROBS)
    parser.add_argument("--datasets", nargs="*", default=["algorithm", "handcrafted"])
    parser.add_argument("--models", nargs="*", default=["gpt-4o-mini", "deepseek-v4-pro"])
    parser.add_argument("--agent-top-n", type=int, default=5)
    parser.add_argument("--algorithm-top-n", type=int, default=5)
    parser.add_argument("--handcrafted-top-n", type=int, default=10)
    parser.add_argument("--prompt-kind", choices=["single", "ascon_SDBL_top", "ascon_llm_prob_history"], default="ascon_llm_prob_history")
    parser.add_argument("--reference-format", choices=["basic", "ranked", "step_probs"], default="basic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-reference-agents", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    prompt_kind_map = {
        "single": "single",
        "ascon_SDBL_top": "ascon",
        "ascon_llm_prob_history": "ascon_llm_prob_history",
    }
    normalized_prompt_kind = prompt_kind_map[args.prompt_kind]

    ranked_records = load_task1_ranked_prob_records(args.algorithm_probs, args.handcrafted_probs)
    dataset_dirs = {
        "algorithm": args.algorithm_dir,
        "handcrafted": args.handcrafted_dir,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    root_summary = {"models": []}
    for model_name in args.models:
        spec = build_model_spec(model_name)
        dataset_summaries = []
        for dataset_name in args.datasets:
            if dataset_name not in dataset_dirs:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            top_n = args.algorithm_top_n if dataset_name == "algorithm" else args.handcrafted_top_n
            summary = evaluate_task1_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dirs[dataset_name],
                top_n=top_n,
                agent_top_n=args.agent_top_n,
                include_agents=args.include_reference_agents,
                prompt_kind=normalized_prompt_kind,
                reference_format=args.reference_format,
                ranked_records=ranked_records,
                output_dir=output_dir,
                spec=spec,
                workers=args.workers,
            )
            dataset_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        root_summary["models"].append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "datasets": dataset_summaries,
                "combined_summary": build_task1_combined_summary(dataset_summaries),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(root_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(root_summary, ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "task1":
        sys.argv.pop(1)
        main_task1()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "task2":
        sys.argv.pop(1)
    main_task2()


if __name__ == "__main__":
    main()
