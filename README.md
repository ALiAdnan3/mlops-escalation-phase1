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
configs/schema.yaml             frozen intents/categories (A)    configs/labels.yaml (B)
params.yaml                     one section per owner; global seed = 42
src/                            one script per pipeline stage
tests/                          schema, leakage, label, pipeline tests
reports/                        validation.json and later metrics
```
