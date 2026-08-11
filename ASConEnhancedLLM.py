from __future__ import annotations

import argparse
import json
import os
import re
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
from ASCon.utils import load_pt

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = ROOT / "Aegis-Bench" / "WWtest_with_agent_error_labels.json"
DEFAULT_INPUT_PT = ROOT / "Aegis-Bench" / "WWtest_with_agent_error_labels.pt"
DEFAULT_CHECKPOINT = ROOT / "results" / "task2_aegis_bench_local" / "seed_42" / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "task2_aegis_enhanced_llm"
DEFAULT_AUX_OUTPUT_JSON = ROOT / "FaultProbabilityFile" / "Task2-WWtest-Agent-fault-probs.json"

ASCON_ENHANCED_AEGIS_COT_PROMPT = """## ROLE AND GOAL
You are a meticulous Multi-Agent System (MAS) Quality Assurance analyst. Your sole purpose is to analyze conversation logs to identify and categorize agent errors based on a strict set of definitions.

## ERROR DEFINITIONS WITH EXAMPLES
You MUST use the exact error codes provided below.

### Functional Mistakes (FM-1.x - Task Execution Errors):
- FM-1.1: **Task specification deviation** - Agent deviates from specified task requirements (e.g., was asked to write code in Python, but used JavaScript).
- FM-1.2: **Role specification deviation** - Agent acts outside its designated role (e.g., a 'CodeWriter' agent starts criticizing other agents' work, which is the 'Critic's' role).
- FM-1.3: **Add redundant steps** - Agent adds unnecessary or duplicate steps (e.g., imports a library that was already imported in a previous step).
- FM-1.4: **Remove conversation history** - Agent ignores or removes important context from previous turns (e.g., ignores a user's correction from the previous message).
- FM-1.5: **Remove termination conditions** - Agent fails to define proper stopping criteria, leading to loops or unfinished tasks (e.g., writes a recursive function with no base case).

### Functional Mistakes (FM-2.x - Communication & Coordination Errors):
- FM-2.1: **Repeat handled tasks** - Agent redundantly handles already completed tasks (e.g., re-writes a piece of code that was already finalized and approved).
- FM-2.2: **Make request ambiguous** - Agent provides unclear or confusing instructions to other agents (e.g., asks another agent to "handle the data" without specifying how).
- FM-2.3: **Deviate from main goal** - Agent pursues objectives unrelated to the main task (e.g., starts discussing the history of programming languages in the middle of a coding task).
- FM-2.4: **Hide important information** - Agent withholds crucial information needed by other agents (e.g., knows a library has a bug but doesn't mention it).
- FM-2.5: **Ignore other agents** - Agent fails to consider input, corrections, or questions from other agents.
- FM-2.6: **Inconsistent reasoning** - Agent's logic contradicts its own previous statements (e.g., in step 2 agent says 'option A is best', but in step 4 says 'option A is a bad choice' without new information).

### Functional Mistakes (FM-3.x - Quality & Verification Errors):
- FM-3.1: **Premature termination** - Agent stops or declares the task complete before all requirements are met.
- FM-3.2: **Remove verification steps** - Agent skips necessary validation or testing steps (e.g., writes code but doesn't write any unit tests for it).
- FM-3.3: **Incorrect verification** - Agent performs flawed or wrong verification (e.g., writes a test that doesn't actually check for the correct condition).

## ANALYSIS WORKFLOW
Please follow these steps carefully:

### Step 1: Agent Summary
First, analyze and summarize what each agent has done throughout the conversation:
- List each agent that appears in the conversation
- For each agent, summarize their main actions, decisions, and contributions
- Note any patterns or recurring behaviors

### Step 2: Error Analysis
For each agent identified in Step 1:
- Carefully examine their actions against each error definition
- Look for violations of task requirements, role boundaries, communication issues, or quality problems
- Note any potential errors with specific reasoning

### Step 3: Final Judgment
Based on your analysis in Steps 1 and 2:
- Determine which agents (if any) committed errors
- Assign the appropriate error code(s) to each faulty agent
- Ensure agent names match exactly as they appear in the conversation log

### AUXILIARY AGENT FAULT PROFILE
Each conversation will be followed by an auxiliary fault profile for every agent, including:
- auxiliary fault probability: the probability that the agent is faulty in this task;
- top 5 likely fault types: the five most likely fault types predicted for that agent.
Please use this auxiliary information to assist your judgment of faulty agents and fault types.

## REQUIRED OUTPUT FORMAT
Your response must contain:

1. **Agent Summary**: A brief analysis of what each agent did
2. **Error Analysis**: Your reasoning for identifying errors
3. **Final Answer**: A valid JSON object with your conclusions

**JSON Format:**
{{"faulty_agents": [{{"agent_name": "XXX", "error_type": "FM-X.X"}}]}}

**Examples:**
- Multiple Errors: {{"faulty_agents": [{{"agent_name": "XXX1", "error_type": "FM-1.1"}}, {{"agent_name": "XXX2", "error_type": "FM-3.2"}}, {{"agent_name": "XXX3", "error_type": "FM-2.5"}}]}}
- No Errors: {{"faulty_agents": []}}

**Important:** Make sure the agent names you output exactly match those in the conversation log. Do not fabricate names.

## CONVERSATION TO ANALYZE:
\"\"\"
{conversation_text}
\"\"\"

## YOUR ANALYSIS:
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        return ModelSpec("deepseek", model_name, os.environ["DEEPSEEK_API_KEY"], disable_thinking=True)
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


def main() -> None:
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


if __name__ == "__main__":
    main()
