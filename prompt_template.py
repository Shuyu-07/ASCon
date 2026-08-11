import json
from typing import Any, Dict, List, Optional


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


SINGLE_ROOT_FAULT_PROMPT = """
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.
The problem is: {problem}.
Identify which agent made an error, at which step, and explain the reason for the error.
Here's the conversation: {chat_content}
Based on this conversation, please predict the following:
    1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.
    2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: {{ "agent a": "x", "agent b": "xxx", "agent c": "xxxx", "agent a": "xxxxxx" }}, each entry represents a step where an agent provides input. If the mistake is in agent c's speech, the step number is 3. If the second speech by agent a contains the mistake, the step number is 4, and so on. Please determine the step number where the first mistake occurred.
    3. The reason for your prediction.
Please answer in the valid JSON format as follows:
    {{
      "agent_name": "Your predicted root-cause faulty agent name",
      "step_number": 1,
      "reason_for_mistake": "Your reason"
    }}
""".strip()


ASCON_ENHANCED_SDBL_ROOT_FAULT_PROMPT = """
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.
The problem is: {problem}.
Identify which agent made an error, at which step, and explain the reason for the error.
Here's the conversation: {chat_content}

The following agents and steps are flagged for special attention. These are the agent and step candidates identified by our automated procedure as the most likely fault locations. Please focus your analysis on these high-confidence candidates and prioritize them when locating the root cause:
{reference_content}

Based on this conversation, please predict the following:
    1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.
    2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: {{ "agent a": "x", "agent b": "xxx", "agent c": "xxxx", "agent a": "xxxxxx" }}, each entry represents a step where an agent provides input. If the mistake is in agent c's speech, the step number is 3. If the second speech by agent a contains the mistake, the step number is 4, and so on. Please determine the step number where the first mistake occurred.
    3. The reason for your prediction.
Please answer in the valid JSON format as follows:
    {{
      "agent_name": "Your predicted root-cause faulty agent name",
      "step_number": 1,
      "reason_for_mistake": "Your reason"
    }}
""".strip()


ASCON_ENHANCED_LLM_ROOT_FAULT_PROMPT = """
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. 
The problem is: {problem}. 
Identify which agent made an error, at which step, and explain the reason for the error. 
Here’s the conversation: {chat_content_with_probility}


Based on this conversation, please predict the following: 
    1. Each step in the conversation is associated with an auxiliary fault probability. Please first examine the high-probability steps and their neighboring context to determine whether they contain the root-cause mistake. 
    2. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert. 
    3. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: {  ”agent a”: ”xx”, ”agent b”: ”xxxx”, ”agent c”: ”xxxxx”, ”agent a”: ”xxxxxxx” },  each entry represents a ’step’ where an agent provides input. The ’x’ symbolizes the speech of each agent. If the mistake is in agent c’s speech, the step number is 3. If the second speech by ’agent a’ contains the mistake, the step number is 4, and so on. Please determine the step number where the first mistake occurred. 
    4. The reason for your prediction.
Please answer in the valid JSON format as follows: 
    {
      "agent_name": "Your predicted root-cause faulty agent name",
      "step_number": 1,
      "reason_for_mistake": "Your reason"
    }
""".strip()


def ordered_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(history, key=lambda item: int(item.get("step_id", 0) or 0))


def build_chat_content(history: List[Dict[str, Any]]) -> str:
    cleaned_history: List[Dict[str, Any]] = []
    for entry in ordered_history(history):
        try:
            step_id = int(entry.get("step_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if step_id == 0:
            continue
        cleaned_history.append({key: value for key, value in entry.items() if key != "is_mistake"})
    return json.dumps(cleaned_history, ensure_ascii=False, indent=2)


def build_chat_content_with_probability(history: List[Dict[str, Any]], ranked_record: Dict[str, Any]) -> str:
    step_probabilities: Dict[int, float] = {}
    for item in ranked_record.get("step_ranked_probabilities", []):
        try:
            step_id = int(item.get("step_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        step_probabilities[step_id] = round(float(item.get("probability", 0.0) or 0.0), 6)

    cleaned_history: List[Dict[str, Any]] = []
    for entry in ordered_history(history):
        try:
            step_id = int(entry.get("step_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if step_id == 0:
            continue
        cleaned_entry = {key: value for key, value in entry.items() if key != "is_mistake"}
        cleaned_entry["fault probability"] = step_probabilities.get(step_id, 0.0)
        cleaned_history.append(cleaned_entry)
    return json.dumps(cleaned_history, ensure_ascii=False, indent=2)


def build_reference_content(
    ranked_record: Dict[str, Any],
    top_n: int,
    agent_top_n: int,
    include_agents: bool,
    reference_format: str = "basic",
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    step_to_agent: Dict[int, str] = {}
    if history:
        for entry in history:
            try:
                history_step_id = int(entry.get("step_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            agent_name = entry.get("name") or entry.get("role")
            if agent_name is not None:
                step_to_agent[history_step_id] = str(agent_name).strip()

    step_ids: List[int] = []
    step_candidates: List[Dict[str, Any]] = []
    for item in ranked_record.get("step_ranked_probabilities", []):
        if len(step_ids) >= top_n:
            break
        step_id = item.get("step_id")
        if step_id is None:
            continue
        try:
            step_id_int = int(step_id)
        except (TypeError, ValueError):
            continue
        if step_id_int == 0:
            continue
        step_ids.append(step_id_int)
        candidate = {
            "step_id": step_id_int,
            "fault_probability": round(float(item.get("probability", 0.0) or 0.0), 6),
        }
        if step_id_int in step_to_agent:
            candidate["agent_name"] = step_to_agent[step_id_int]
        step_candidates.append(candidate)

    agents: List[str] = []
    agent_candidates: List[Dict[str, Any]] = []
    if include_agents:
        for item in ranked_record.get("agent_ranked_probabilities", []):
            if len(agents) >= agent_top_n:
                break
            agent_name = item.get("agent_name")
            if agent_name is None:
                continue
            agent_name_text = str(agent_name).strip()
            if not agent_name_text or agent_name_text.lower() in {"user", "human"}:
                continue
            agents.append(agent_name_text)
            agent_candidates.append(
                {
                    "agent_name": agent_name_text,
                    "fault_probability": round(float(item.get("probability", 0.0) or 0.0), 6),
                }
            )

    if reference_format == "step_probs":
        return json.dumps({"ranked_step_candidates": step_candidates}, ensure_ascii=False, indent=2)
    if reference_format == "ranked":
        return json.dumps(
            {
                "step_candidates": step_candidates,
                "agent_candidates": agent_candidates,
            },
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(
        {
            "step_ids": step_ids,
            "agents": agents,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_task1_prompt(
    problem: str,
    chat_content: str,
    reference_content: str,
    prompt_kind: str,
    chat_content_with_probability: str = "",
) -> str:
    if prompt_kind == "ascon_llm_prob_history":
        return (
            ASCON_ENHANCED_LLM_ROOT_FAULT_PROMPT
            .replace("{problem}", problem)
            .replace("{chat_content_with_probility}", chat_content_with_probability)
        )
    if prompt_kind == "single":
        return (
            SINGLE_ROOT_FAULT_PROMPT
            .replace("{problem}", problem)
            .replace("{chat_content}", chat_content)
        )
    return (
        ASCON_ENHANCED_SDBL_ROOT_FAULT_PROMPT
        .replace("{problem}", problem)
        .replace("{chat_content}", chat_content)
        .replace("{reference_content}", reference_content)
    )
