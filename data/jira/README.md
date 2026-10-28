# Apache Jira benchmark data

Source: Apache Jira issue-tracking dataset of Diamantopoulos, Nastos and Symeonidis (MSR 2023),
https://doi.org/10.5281/zenodo.14253918, CC BY 4.0. Issue text is **not** included; rebuild it with
`python -m benchmarks.jira.build_issues` from `issues.bson.gz` and `events.bson.gz` (see the main README).
All fields are the creation-time values (build `v1.1`).

| File | Contents |
|---|---|
| `sample_issue_ids.txt` | The 29,997 sampled issue identifiers (`issue_id`, in time order) |
| `labels_splits_features.csv.gz` | One row per sampled issue: creation-time priority label, time split, context features (below) |
| `d1_qwen25_7b.jsonl` | D1 descriptors (Qwen2.5-7B-Instruct) for 29,988 of the 29,997 issues (9 have no valid D1 record) |
| `projects.csv`, `build_stats.json` | Project selection and reconstruction statistics of the full dump |

Columns of `labels_splits_features.csv.gz` (gzip, mtime 0; regenerate or verify with
`python -m benchmarks.jira.export_release_data [--check]`):

| Column | Meaning |
|---|---|
| `issue_id`, `issue_key`, `project`, `type`, `created` | Identifiers and creation-time metadata (`created` in UTC) |
| `priority_creation` | Priority at creation (Blocker, Critical, Major, Minor, Trivial) |
| `label3` | `High` = Blocker/Critical, `Medium` = Major, `Low` = Minor/Trivial |
| `s2_non_default` | 1 if the creation priority is not Major (membership of the S2 sensitivity setting) |
| `split` | Time split by `created`: `train` < 2022-01-01, `val` in 2022, `test` >= 2023-01-01 |
| `reporter_prior_n`, `comp_prior_n` | Number of earlier project-selected issues by the same reporter / with the same component string |
| `reporter_prior_high_share`, `reporter_prior_low_share` | Share of High / Low labels among the reporter's earlier issues (0 if none) |
| `desc_chars`, `summary_chars`, `h_n_components`, `h_has_attachment` | Description and summary length, number of components, attachment flag at creation |

The history features are computed on the full project-selected table (all issues strictly earlier
in time) before the sample is drawn, so they cannot be recomputed from the sample alone; they are
shipped as computed. Reporter and component names are not distributed.
