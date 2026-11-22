# mlops-escalation-phase1

Phase 1 of an autonomous customer-support system: given a customer query and a knowledge base of
support answers, **retrieve** the best answers and **predict whether to escalate** to a human.
This phase proves the ML pipeline is versioned, leak-free and reproducible.

Pipeline (DVC): `ingest → validate → split → build_kb → baseline → train → evaluate → register`

## Quick start

```bash
git clone https://github.com/ALiAdnan3/mlops-escalation-phase1.git
cd mlops-escalation-phase1
python -m venv .venv            # Python 3.11
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock   # exact versions (requirements.txt = top-level pins)
dvc repro                          # rebuilds everything; ingest downloads from Hugging Face, no account needed
pytest
```

## Data

- **Source:** [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
  pinned to commit `430d1a89bd93bd1fa23c16f29dd53e73f0087443` (`params.yaml → ingest.hf_revision`).
- **Verified contents:** 26,872 rows, 27 intents, 11 categories (frozen in `configs/schema.yaml`).
- `data/raw/bitext.parquet` is the untouched download; `data/interim/validated.parquet` adds a stable
  integer `row_id` (row position in the pinned raw file).

## Validation gate (`validate` stage)

`src/validate.py` runs a Pandera schema and exits non-zero on any failure, so `dvc repro` stops before
training. It always writes `reports/validation.json` (pass/fail, each failed rule with examples, dataset stats).

| Rule | Check |
| --- | --- |
| `required_columns` | `instruction`, `response`, `category`, `intent`, `flags` exist |
| `not_nullable`, `is_string`, dtype | no nulls; text columns hold strings |
| `not_blank`, `length_bounds` | no empty/whitespace text; instruction 3–500 chars, response 3–3000 chars |
| `known_values` | intent in the 27 known intents, category in the 11 known categories |
| `intent_category_consistent` | each intent sits in its own category |
| `flag_letters` | flags are upper-case letter codes |
| `duplicate_rate` | exact `(instruction, intent)` duplicates ≤ 10% (observed 8.3%: queries are templated) |
| `min_rows_per_intent`, `min_total_rows` | every intent has ≥ 500 rows; ≥ 20,000 rows overall |

Thresholds live in `params.yaml → validate`. Tests: `pytest tests/test_validate.py` feeds deliberately
corrupted copies of a 54-row real fixture (missing column, null/unknown intent, unknown category, empty
text, wrong dtype, overlong text, intent/category mismatch, duplicate flood, missing intent, bad flags)
and asserts each fails with a clear message, and that the stage exits non-zero without writing output.

## Labels, leak-free splits, KB and baseline (Member B)

**Labels** (`configs/labels.yaml`, frozen): `label_escalate` is a rule on the Bitext intent.
R1 human/complaint (`contact_human_agent`, `complaint`), R2 high-risk (`payment_issue`, `delete_account`),
R3 out of KB (`switch_account`, `newsletter_subscription`, `review`), R0 everything else. Positive rate 26.0%.

**Split** (`split` stage, `src/split.py`). Bitext queries are templated, so near-identical twins exist
("i want assistance canceling order" / "i need assistance canceling order"); a random row split would leak
15.5% of test queries as exact copies of train queries. Instead:
1. normalise queries (lowercase, drop `{{placeholders}}`, digits, punctuation);
2. within each intent, link queries whose TF-IDF character n-gram cosine is >= 0.85 and take connected
   components as `group_id` (threshold justified in `reports/eda.md`);
3. assign whole groups to train/val/test (70/15/15) with a greedy per-intent fill, so every intent and the
   positive rate (26.0%) are balanced in every split.

**KB** (`build_kb` stage, `src/kb.py`): response texts of **train rows only**, R3 intents removed (so those
queries have no answer and should escalate), de-duplicated, capped at 50 per intent: 1,200 entries.

**Leakage evidence** (`pytest tests/test_split.py`): no `group_id` shared across splits; no identical
normalised query across splits; no same-intent pair with similarity >= 0.85 across splits; every KB entry is a
train row; no R3 intent in the KB; labels equal the YAML rules; balanced shares; same seed gives identical files.

**Baseline, run #0** (`baseline` stage, `src/baseline.py`): BM25 over the KB; escalate if the query contains
an explicit human/complaint keyword or the top-1 BM25 score per query token is below tau (tuned on val only).
Metrics (shared code in `src/metrics.py`): intent_hit_at_1/5 and MRR on in-KB queries; escalation precision,
recall, F1, PR-AUC on all queries. Written to `reports/baseline_metrics.json` and logged to MLflow as `baseline-bm25`.

**MLflow credentials.** Runs go to DagsHub when `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are set (environment or
a git-ignored `.env`, see `.env.example`). Without them the stage logs to a local `sqlite:///mlflow.db`, so
graders can run `dvc repro` with no account. All team runs used for submission must be on DagsHub.

## DVC setup (team members)

Data is tracked by DVC (`dvc.yaml` + `dvc.lock` in git, file contents in the DagsHub remote).

**One-time remote setup.** Credentials go in `.dvc/config.local`, which git ignores. Never commit a token.

```bash
# committed config (already in .dvc/config once the owner has run it):
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<dagshub-owner>/mlops-escalation-phase1.s3

# each member, locally only:
dvc remote modify origin --local access_key_id     <your-dagshub-token>
dvc remote modify origin --local secret_access_key <your-dagshub-token>
```

**Daily commands**

```bash
git pull && dvc pull                 # get code + matching data
dvc repro <stage>                    # rerun a stage and anything upstream that changed
dvc status                           # what is out of date
dvc dag                              # show the pipeline
git add dvc.lock <changed files> && git commit -m "..."   # commit dvc.lock WITH the code that produced it
dvc push                             # upload data outputs to the remote
```

**Handoff:** commit, `dvc push`, then post the commit hash and file path in the team chat. The receiver
runs `git pull && dvc pull` and confirms.

Graders do not need DagsHub access: `dvc repro` regenerates all data from the pinned Hugging Face revision.

## Repo layout

```
data/{raw,interim,processed}/   DVC-tracked stage outputs
configs/schema.yaml             frozen intents/categories (A)
configs/labels.yaml             frozen escalation rules R0-R3 (B)
params.yaml                     one section per owner; global seed = 42
src/                            one script per pipeline stage
tests/                          schema, leakage, label, pipeline tests
reports/                        validation.json, eda.md, split_report.json, baseline_metrics.json
```
