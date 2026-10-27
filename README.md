# LMEIC: LLM-driven Multi-source Email Importance Classification

Code, prompts, configurations and public-benchmark data for the paper

> **A multi-source LLM-driven framework for recipient-aware enterprise email importance classification**
> Xingyun Yan, Jie Hu, Qingke Jia, Yibo Huang, Lingyu Wang, Jin Qi, Bin Chen, Mingzhu Fang, Meixing Guo, Hairui Wang, Senyou Shi.
> Submitted to *International Journal of Machine Learning and Cybernetics* (under revision).

LMEIC classifies an email-recipient instance as High, Medium or Low importance. A large language
model extracts a fixed set of structured descriptors from the email text (content domain, D1);
relational (D2) and role-context (D3) features are computed from strictly pre-timestamp history;
a lexical channel and a cross-fitted deep-text co-reader add text evidence; and a stacking layer
(XGBoost + LightGBM + logistic regression) assigns the label. The LLM never assigns the label.

## What is in this repository

| Path | Contents |
|---|---|
| `lmeic/features/` | D1 extraction (API and local vLLM), D2/D3 features with pre-timestamp history, extended context features, privacy masking |
| `lmeic/reference/` | Feature assembly, thread-grouped tuning and stacking, co-reader cross-fitting, paired thread-clustered bootstrap analysis, main-table script |
| `lmeic/analysis/` | Domain ablations and context subsets, SHAP by domain, predictive / interaction information, redundancy-reduction architectures, extractor perturbation, robustness checks |
| `lmeic/labelling/` | Multi-agent labelling prompts and protocol (three agents, two-of-three consensus, one debate round, adjudication) |
| `lmeic/human_study/` | Inter-annotator agreement, alternative annotator test, re-scoring on human labels, output-layer recalibration |
| `lmeic/baselines/` | LoRA-tuned Qwen2.5-7B classifier and open-weight zero-shot baselines |
| `lmeic/deploy/` | Latency, deployment comparison with the LoRA classifier, monthly retraining replay |
| `benchmarks/enron/` | Recipient-aware Enron benchmark: instance construction, context features, models, LoRA comparison |
| `benchmarks/jira/` | Apache Jira creation-time priority benchmark: priority and field reconstruction, baselines, D1 fusion, identity ablation |
| `configs/` | Experiment configurations (hyperparameter search spaces, seeds, prompts, role descriptions) |
| `figures/` | Scripts for the manuscript figures |
| `data/` | Derived data for the two public benchmarks (see below) |

## Data

**EMAIL-3R** (the main corpus) is an enterprise email corpus that is confidential. Its messages,
labels, derived features and trained model weights are not distributed. Organisation-specific
configuration is replaced by templates (`configs/private/partner_identifiers.example.yaml`), and
the email reply-boundary parser used in preprocessing is not included.

**Public benchmarks** (`data/`), released under CC BY 4.0:

| File | Contents |
|---|---|
| `data/enron/instances.jsonl.gz` | 5,140 email-recipient instances: mailbox owner, role (EXE / TRD / STF), timestamp, thread, split, and `email_key` (first 16 hex digits of the SHA-1 of sender, date, normalised subject and the first 200 body characters, as computed by `build_instances.py`) |
| `data/enron/agent_labels.jsonl` | Agent-consensus labels and round-zero votes for 5,137 instances |
| `data/enron/d1_gpt4o.jsonl` | D1 descriptors extracted with `gpt-4o-2024-11-20` |
| `data/enron/context_features.parquet` | D2/D3 features computed from each owner's pre-timestamp history |
| `data/jira/sample_issue_ids.txt` | The 29,997 sampled issue identifiers |
| `data/jira/d1_qwen25_7b.jsonl` | D1 descriptors extracted with Qwen2.5-7B-Instruct from the creation-time summary and description |
| `data/jira/projects.csv`, `build_stats.json` | Project selection and reconstruction statistics |

Email bodies and issue texts are not redistributed. Rebuild them from the original sources:

- Enron corpus: <https://www.cs.cmu.edu/~enron/> (we used the Hugging Face copy `corbt/enron-emails`); then run `python -m benchmarks.enron.build_instances`.
- Apache Jira issue-tracking dataset (Diamantopoulos et al., MSR 2023), CC BY 4.0: <https://doi.org/10.5281/zenodo.14253918>; download `issues.bson.gz` and `events.bson.gz`, then run `python -m benchmarks.jira.build_issues`.

## Setup

```bash
conda create -n lmeic python=3.11 -y && conda activate lmeic
pip install -r requirements.txt
# local extraction and zero-shot baselines use vLLM in a separate environment
pip install vllm==0.32.0
```

API keys are read from environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`). Paths are set in `configs/paths.yaml`.

## Reproducing the public-benchmark results

```bash
# Apache Jira
python -m benchmarks.jira.build_issues
python -m benchmarks.jira.run_baselines
python -m benchmarks.jira.run_d1_fusion --d1 data/jira/d1_qwen25_7b.jsonl --setting S1
python -m benchmarks.jira.run_d1_fusion --d1 data/jira/d1_qwen25_7b.jsonl --setting S2
python -m benchmarks.jira.run_id_ablation --d1 data/jira/d1_qwen25_7b.jsonl

# Enron
python -m benchmarks.enron.build_instances
python -m benchmarks.enron.run_models --run-dir runs/enron --labels data/enron/agent_labels.jsonl --d1 data/enron/d1_gpt4o.jsonl --d1-name gpt4o --context data/enron/context_features.parquet
```

The EMAIL-3R scripts in `lmeic/` document the exact protocol used in the paper (thread-grouped
five-fold tuning, five seeds, thread-clustered paired bootstrap with Holm correction) and can be
applied to any corpus with the same record schema.

## Citation

See `CITATION.cff`. The citation will be updated with the journal reference on publication.

## Licence

Code: MIT (`LICENSE`). Derived benchmark data in `data/`: CC BY 4.0 (`data/LICENSE`). The Enron
corpus and the Apache Jira dataset remain under their original terms.
