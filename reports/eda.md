# EDA: labels, duplicates and the grouping threshold (Member B)

Data: `data/interim/validated.parquet`, 26,872 rows, 27 intents, 11 categories (Bitext, HF revision `430d1a89bd`).

## Labels

| Rule | Intents | Rows |
| --- | --- | ---: |
| R1 explicit human / complaint | contact_human_agent, complaint | 1,999 |
| R2 high-risk | payment_issue, delete_account | 1,994 |
| R3 out of KB | switch_account, newsletter_subscription, review | 2,996 |
| R0 bot answers | the other 20 intents | 19,883 |

**Positive rate (`label_escalate = 1`): 26.0%** (6,989 / 26,872). Intents are near-balanced: 950 to 1,000 rows each.

## Why the split must be by group

- Queries are generated from templates with injected typos and placeholders, for example
  `i need help cancelling puchase {{Order Number}}`, `i have a question about cancelling oorder {{Order Number}}`.
- 24.8% of queries contain a `{{placeholder}}` (`{{Order Number}}` alone appears 2,907 times).
- Exact duplicate queries: 8.3%. After normalisation (lowercase, drop placeholders, digits and punctuation): **12.3%**.
- No query text appears under two different intents.
- Responses are almost all unique (26,870 of 26,872), so de-duplicating responses barely shrinks the KB; `kb_per_intent` does.
- The median query's nearest same-intent neighbour has character n-gram cosine **0.94**: almost every query has a near-twin.
- **A random row split leaks**: 15.5% of test queries would have an exact normalised copy in train.

## Choosing `split.group_threshold`

Method: TF-IDF character n-grams (`char_wb`, 3-5) on the normalised query, cosine similarity within each intent,
connected components of the graph `similarity >= threshold` become `group_id`.
Character n-grams are used because they still match through typos ("cancelling" / "canceling" / "cancelation").

Example same-intent pairs per similarity band:

| Band | Example pair |
| --- | --- |
| 0.80-0.85 | `wanna chat with a person` / `i want to chat with a person` |
| 0.85-0.90 | `can you help me to talk to a human agent` / `can ya help me talk with a human agent` |
| 0.90-0.95 | `i want assistance canceling order` / `i need assistance canceling order` |
| 0.95-1.00 | `want assistance cancelling order` / `i want assistance cancelling order` |

Threshold sweep (connected components; group size = rows):

| Threshold | Groups | Largest groups | Effect |
| ---: | ---: | --- | --- |
| 0.60 | 303 | 999, 998, 997 | whole intents become one group (chaining): unusable |
| 0.70 | 1,140 | 980, 970, 967 | still whole intents |
| 0.80 | 4,798 | 826, 732, 675 | groups of most of an intent; per-intent val share 9-28% |
| **0.85** | **7,976** | **299, 250, 186** | **chosen** |
| 0.90 | 12,585 | 151, 133, 125 | balanced, but lets 0.85-0.90 template twins cross splits |

**0.85 is the lowest threshold whose groups stay well below intent size.** Pairs at or above 0.85 are the same
template with one word changed, so they must not be split; connected components guarantee that no same-intent pair
at or above 0.85 crosses splits (checked by `tests/test_split.py::test_no_near_duplicate_pair_crosses_splits`).

## Split result (seed 42)

Groups are assigned with a greedy per-intent fill (biggest groups first, each into the split furthest below its
70/15/15 target). `StratifiedGroupKFold` was tried first: at 0.85 it gave one intent only 2% test rows and positive
rates of 20% (val) vs 28% (train), because large groups make fold packing coarse.

| Split | Rows | Share | Positive rate | Per-intent share range |
| --- | ---: | ---: | ---: | --- |
| train | 18,809 | 70.0% | 26.0% | 69.9-70.1% |
| val | 4,035 | 15.0% | 26.0% | 15.0-15.1% |
| test | 4,028 | 15.0% | 26.0% | 14.9-15.0% |

KB: 1,200 entries = 24 in-KB intents x 50 (`kb_per_intent`), train rows only, R3 intents excluded.

## Baseline design notes

Keyword precision on **train** (share of matching queries that should escalate), used to pick `baseline.keyword_regex`:

| Keyword | Matches | Precision | Kept |
| --- | ---: | ---: | --- |
| agent / operator / human / chat / assistant | 96-279 each | 1.00 | yes |
| complain / claim | 231 / 282 | 1.00 | yes |
| someone | 61 | 0.92 | yes |
| person | 677 | 0.12 | no (`{{Person Name}}` placeholder, invoice queries) |
| talk / speak | 342 / 379 | 0.61 / 0.72 | no (also `contact_customer_service`) |
| newsletter / review / feedback | | 1.00 | no: these restate the R3 label rule; R3 is left to the BM25 threshold |

Retrieval confidence for out-of-KB (R3) detection, val, non-keyword queries (ROC-AUC, 0.5 = chance):
raw top-1 BM25 0.575, top-1 minus top-2 0.569, **top-1 per query token 0.647** (used).
Retrieval scores alone separate R3 poorly, which is why the primary model adds query features.

## Known limits

- The data is templated, so even after grouping a test query usually has a train query in the 0.80-0.85 band.
  Grouping removes trivial copies; it cannot make a templated dataset behave like free-form customer text.
- Labels are a rule on the intent, so they are only as correct as Bitext's intent annotation.
