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
| `lmeic/human_study/` | Inter-annotator agreement, alternative annotator test, re-scoring on human labels, output-layer recalibration. The human-annotation data are not released, so these scripts document the analysis but cannot be run without them |
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
| `data/jira/labels_splits_features.csv.gz` | Per issue: creation-time priority, 3-class label, time split (train < 2022, val 2022, test >= 2023) and the eight context features of the baselines; no issue text (column list in `data/jira/README.md`) |
| `data/jira/d1_qwen25_7b.jsonl` | D1 descriptors extracted with Qwen2.5-7B-Instruct from the creation-time summary and description (29,988 issues) |
| `data/jira/projects.csv`, `build_stats.json` | Project selection and reconstruction statistics |

The Jira labels, splits and context features are derived from the Apache Jira dataset (CC BY 4.0,
see `data/LICENSE` for the attribution). The Jira benchmark uses the creation-time priorities set by
human reporters, so it has no agent labels.

Email bodies and issue texts are not redistributed. Rebuild them from the original sources:

- **Enron.** Put the Hugging Face copy of the corpus (`corbt/enron-emails`, <https://huggingface.co/datasets/corbt/enron-emails>; original corpus <https://www.cs.cmu.edu/~enron/>) at `datasets/public/enron/corbt_enron_emails/data/train-*.parquet`, and the mailbox-owner position list at `datasets/public/enron/positions/E-Mails-updated.csv` (paths in `configs/rev1_e12_enron.yaml`). `python -m benchmarks.enron.build_instances` joins the two and rebuilds the instances **with** `subject` and `latest_text`; the construction is seeded, and `--check data/enron/instances.jsonl.gz` verifies that the rebuilt `instance_id`, `email_key`, owner, role, date, split and hashed thread id equal the shipped list. The shipped `instances.jsonl.gz` itself has no text and cannot be used by `run_models` directly.
  *Known gap:* `E-Mails-updated.csv` (semicolon-separated; columns `eid;firstName;lastName;Ordner;status;E-Mail1;E-Mail2;E-Mail3`) is not redistributed here. The configuration records that it was taken from the GitHub project `ahr85/enron`, which reconstructs the employee list of Shetty and Adibi (2004); its exact path and licence are not recorded in this repository, and an audit against the original ISI table found that it is not a verbatim copy (67 of the 82 mailbox owners have the same position class; `benchmarks/enron/audit_positions.py`). Without this file the Enron instances cannot be rebuilt.
- **Apache Jira.** Download `issues.bson.gz` and `events.bson.gz` from <https://doi.org/10.5281/zenodo.14253918> (CC BY 4.0) into `datasets/public/apache_jira/raw/`, then run `python -m benchmarks.jira.build_issues` (output: `datasets/public/apache_jira/processed/issues.parquet`, with text, fields at creation time). `python -m benchmarks.jira.export_release_data --check` verifies that the rebuilt labels, splits and features equal the shipped `labels_splits_features.csv.gz`.

## Setup

```bash
conda create -n lmeic python=3.11 -y && conda activate lmeic
pip install -r requirements.txt
# local extraction and zero-shot baselines use vLLM in a separate environment
pip install vllm==0.32.0
```

API keys are read from environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`). Paths are set in `configs/paths.yaml`.

## Reproducing the public-benchmark results

Commands are run from the repository root. Paths follow `configs/paths.yaml` and
`datasets/public/...` as above.

```bash
# Apache Jira (needs datasets/public/apache_jira/raw/{issues,events}.bson.gz, see Data)
python -m benchmarks.jira.build_issues                      # -> datasets/public/apache_jira/processed/
python -m benchmarks.jira.export_release_data --check       # optional: rebuilt labels/splits/features == shipped
python -m benchmarks.jira.run_baselines                     # draws the same 29,997-issue sample as data/jira/sample_issue_ids.txt
python -m benchmarks.jira.run_d1_fusion --d1 data/jira/d1_qwen25_7b.jsonl --setting S1
python -m benchmarks.jira.run_d1_fusion --d1 data/jira/d1_qwen25_7b.jsonl --setting S2
python -m benchmarks.jira.run_id_ablation --d1 data/jira/d1_qwen25_7b.jsonl
```

`--sample-ids` defaults to the shipped `data/jira/sample_issue_ids.txt`; results go to timestamped
directories under `datasets/public/apache_jira/processed/`.

```bash
# Enron (needs the corpus and the positions file, see Data)
python -m benchmarks.enron.build_instances --out-dir runs/enron --check data/enron/instances.jsonl.gz
python -m benchmarks.enron.run_models --run-dir runs/enron --labels data/enron/agent_labels.jsonl \
    --d1 data/enron/d1_gpt4o.jsonl --d1-name gpt4o --context data/enron/context_features.parquet
```

Without `--out-dir`, `build_instances` writes to a new timestamped directory `runs/rev1_e12_enron_<timestamp>/`.
**The Enron command above reproduces only the explicit LMEIC variants** (D1 + lexical + D2/D3 + interactions);
`run_models` silently drops every variant that needs the co-reader. **LMEIC-full** (explicit variants plus the
cross-fitted XLM-R co-reader) needs one additional GPU step before `run_models`:

```bash
python -m benchmarks.enron.coreader --run-dir runs/enron --labels data/enron/agent_labels.jsonl   # CUDA GPU; writes runs/enron/coreader_<timestamp>/
python -m benchmarks.enron.run_models --run-dir runs/enron --labels data/enron/agent_labels.jsonl \
    --d1 data/enron/d1_gpt4o.jsonl --d1-name gpt4o --context data/enron/context_features.parquet \
    --coreader runs/enron/coreader_<timestamp>
```

Tuning uses Optuna with five seeds and 2,000 bootstrap draws and takes hours on CPU. Re-extracting the D1
descriptors (`benchmarks/jira/extract_d1.py`, `lmeic/features/`) needs a GPU with vLLM or an LLM API key;
the shipped descriptor files make this step optional.

The EMAIL-3R scripts in `lmeic/` document the exact protocol used in the paper (thread-grouped
five-fold tuning, five seeds, thread-clustered paired bootstrap with Holm correction) and can be
applied to any corpus with the same record schema. EMAIL-3R itself is confidential: the input paths in
these scripts and in `configs/rev1_*.yaml` (for example `private/runs/...`, `v1_experiments/...`) are
placeholders for locations of the private data. The scripts in `lmeic/human_study/` additionally need
the human-annotation data, which are not released, so they cannot be run from this repository.

## Citation

See `CITATION.cff`. The citation will be updated with the journal reference on publication.

## Licence

Code: MIT (`LICENSE`). Derived benchmark data in `data/`: CC BY 4.0 (`data/LICENSE`, with attribution to the Enron corpus
and to the Apache Jira dataset of Diamantopoulos et al., CC BY 4.0). The Enron corpus remains under its
original terms.
