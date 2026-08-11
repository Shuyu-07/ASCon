# ASCon

This directory contains the public implementation of **ASCon: A Direction-Aware Reciprocal Agent-Step Contextualization Model for Failure Attribution in Multi-Agent Systems**.

ASCon studies two representative MAS failure-attribution settings:

- **Task 1: root-fault attribution**, which identifies one root responsible agent and one key failure step.
- **Task 2: failure-mode attribution**, which identifies faulty agents and their corresponding failure modes.

Accordingly, ASCon uses two prediction heads for these two settings.

As we do not hold the rights to publicly release the `TracerTraj` dataset, we use Task 2 on Aegis-Bench as the example here to demonstrate how to train and use the ASCon model.

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

### (3) Use the trained model predictions to enhance the LLM

This step has two sub-steps:

1. use the trained ASCon Task 2 model to generate the auxiliary agent fault-probability file;
2. use that probability file to enhance the LLM input.

Run:

```bash
set OPENAI_API_KEY=your_key_here
python -m ASCon.ASConEnhancedLLM ^
  --input_json ASCon/Aegis-Bench/WWtest_with_agent_error_labels.json ^
  --input_pt ASCon/Aegis-Bench/WWtest_with_agent_error_labels.pt ^
  --checkpoint ASCon/results/task2_aegis_bench_local/seed_42/best.pt ^
  --aux_output_json ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json ^
  --model gpt-4o-mini ^
  --workers 4
```

This command first generates:

- `ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json`

and then uses this probability file to enhance the LLM reasoning on `WWtest`.

Example with `deepseek-v4-pro`:

```bash
set DEEPSEEK_API_KEY=your_key_here
python -m ASCon.ASConEnhancedLLM ^
  --input_json ASCon/Aegis-Bench/WWtest_with_agent_error_labels.json ^
  --input_pt ASCon/Aegis-Bench/WWtest_with_agent_error_labels.pt ^
  --checkpoint ASCon/results/task2_aegis_bench_local/seed_42/best.pt ^
  --aux_output_json ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json ^
  --model deepseek-v4-pro ^
  --workers 4
```

The LLM evaluation results are written under:

- `ASCon/results/task2_aegis_enhanced_llm/<model>/records.jsonl`
- `ASCon/results/task2_aegis_enhanced_llm/<model>/summary.json`

## Reference Probability Files

We also provide several reference probability samples in:

- `ASCon/FaultProbabilityFile`

For example:

- `ASCon/FaultProbabilityFile/Task1-WWtest-Algorithm_ranked_probs.json`
- `ASCon/FaultProbabilityFile/Task1-WWtest-Handcraft_ranked_probs.json`
- `ASCon/FaultProbabilityFile/Task2-WWtest-Agent-fault-probs.json`

## Task 2 Metrics

Task 2 reports:

- `faulty_agent`: micro/macro F1
- `error_mode`: micro/macro F1
- `agent_error_pair`: micro/macro F1
