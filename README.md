# ASCon

This directory contains the public implementation of **ASCon: A Direction-Aware Reciprocal Agent-Step Contextualization Model for Failure Attribution in Multi-Agent Systems**.

ASCon is conducted on the following two representative MAS failure-attribution settings:

- **Task 1: root-fault attribution**, which identifies one root responsible agent and one key failure step.
- **Task 2: failure-mode attribution**, which identifies faulty agents and their corresponding failure modes.

Here, we use Task 2 on Aegis-Bench as the example here to demonstrate how to train and use the ASCon model.

### (1) Run the preprocessing script to build encoded `pt` files

Run:

```bash
python -m ASCon.build_aegis_bench_pt
```

This script reads:

- `ASCon/Aegis-Bench/train_with_agent_error_labels.json`
- `ASCon/Aegis-Bench/val_with_agent_error_labels.json`
- `ASCon/Aegis-Bench/test_with_agent_error_labels.json`
- `ASCon/Aegis-Bench/WWtest_with_agent_error_labels.json`

It writes:

- `ASCon/Aegis-Bench/train_with_agent_error_labels.pt`
- `ASCon/Aegis-Bench/val_with_agent_error_labels.pt`
- `ASCon/Aegis-Bench/test_with_agent_error_labels.pt`
- `ASCon/Aegis-Bench/WWtest_with_agent_error_labels.pt`
- `ASCon/Aegis-Bench/agent_error_type_to_index.json`

### (2) Run the model training script

Run:

```bash
python -m ASCon.train_task2 ^
  --data_dir ASCon/Aegis-Bench ^
  --output_dir ASCon/results/task2_aegis_bench_local ^
  --seed 42 ^
  --epochs 12
```

The trained best model is saved to:

- `ASCon/results/task2_aegis_bench_local/seed_42/best.pt`

### (3) Use ASCon predictions to enhance LLM-based attribution

ASCon-enhanced LLM evaluation covers both attribution settings:

- **Task 1: root-fault attribution**
  - ASCon ranked fault candidates are used to enhance root-cause localization on `WhoWhen-RootTest`.
  - For the SDBL-style setting implemented here, the LLM receives ASCon-ranked fault evidence from:
    - `ASCon/FaultProbabilityFile/Task1-WWtest-Algorithm_ranked_probs.json`
    - `ASCon/FaultProbabilityFile/Task1-WWtest-Handcraft_ranked_probs.json`
  - The supported prompt modes are:
    - `single`: no ASCon auxiliary evidence
    - `ascon_SDBL_top`: append ASCon top-ranked step / agent candidates as reference content. Used for ASCon-enhanced-SDBL method.
    - `ascon_llm_prob_history`: append ASCon step fault probabilities directly into the conversation history

- **Task 2: failure-mode attribution**
  - The trained Task 2 model first generates an auxiliary agent fault profile for `WWtest`.
  - The profile contains each agent's fault probability and top-five candidate fault types.
  - This profile is then appended to the LLM input for failure-mode attribution.

**Task 1 example**

```bash
set OPENAI_API_KEY=your_key_here
python -m ASCon.ASConEnhancedLLM task1 ^
  --model gpt-4o-mini ^
  --prompt-kind ascon_llm_prob_history ^
  --workers 4
```

Task 1 reads:

- `ASCon/WhoWhen-RootTest/Algorithm-Generated`
- `ASCon/WhoWhen-RootTest/Hand-Craft`
- `ASCon/FaultProbabilityFile/Task1-WWtest-Algorithm_ranked_probs.json`
- `ASCon/FaultProbabilityFile/Task1-WWtest-Handcraft_ranked_probs.json`

Task 1 results are written under:

- `ASCon/results/task1_root_enhanced_llm/<dataset>/<provider>/<model>/records.jsonl`
- `ASCon/results/task1_root_enhanced_llm/<dataset>/<provider>/<model>/summary.json`
- `ASCon/results/task1_root_enhanced_llm/summary.json`

**Task 2 example**

```bash
set OPENAI_API_KEY=your_key_here
python -m ASCon.ASConEnhancedLLM task2 ^
  --input_json ASCon/Aegis-Bench/WWtest_with_agent_error_labels.json ^
  --input_pt ASCon/Aegis-Bench/WWtest_with_agent_error_labels.pt ^
  --checkpoint ASCon/results/task2_aegis_bench_local/seed_42/best.pt ^
  --aux_output_json ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json ^
  --model gpt-4o-mini ^
  --workers 4
```

This command first generates:

- `ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json`

and then uses this profile to enhance the LLM reasoning on `WWtest`.

Task 2 results are written under:

- `ASCon/results/task2_aegis_enhanced_llm/<model>/records.jsonl`
- `ASCon/results/task2_aegis_enhanced_llm/<model>/summary.json`

## Reference Probability Files

We also provide several reference probability samples in:

- `ASCon/FaultProbabilityFile`

For example:

- `ASCon/FaultProbabilityFile/Task1-WWtest-Algorithm_ranked_probs.json`
- `ASCon/FaultProbabilityFile/Task1-WWtest-Handcraft_ranked_probs.json`
- `ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json`

## 📚 Datasets and Acknowledgements

Our experiments are built upon the following failure-attribution datasets and benchmarks. We sincerely thank the authors of these works for making their valuable resources available to the research community.

* **TracerTraj** — introduced in *AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? (ICLR 2026)*

* **Aegis-Bench** — introduced in *Aegis: Automated Error Generation and Attribution for Multi-Agent Systems (ICLR 2026)*.

* **Who&When** — introduced in *Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems (ICML 2025)*.


We gratefully acknowledge the authors of these works for providing the datasets used in our experiments.

### Data Availability
The expanded TracerTraj dataset is not redistributed in this repository, as we do not have permission to publicly redistribute the dataset. Researchers interested in obtaining TracerTraj are kindly referred to the original authors or the official source of the dataset.

---

## 📝 Citation

If you find ASCon useful in your research, please consider citing our paper:

```bibtex
@inproceedings{TODO,
  title     = {ASCon: A Direction-Aware Reciprocal Agent--Step Contextualization Model for Failure Attribution in Multi-Agent Systems},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

---
