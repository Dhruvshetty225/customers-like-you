# Customers Like You: GiftLane recommender & customer map

Compress GiftLane's customer x product purchase matrix with **Truncated SVD** so that:
(1) the "Customers who bought this also bought…" box can be rebuilt every night, and
(2) marketing gets one picture of the main customer groups.

**Team / roles:** Dhruv Shetty, Data Engineer + ML Engineer (individual project)

## Success targets (J1, agreed Day 1)

| # | Target | How we measure it | Result |
|---|--------|-------------------|--------|
| 1 | Compressed data is much smaller than the full matrix | stored factors MB vs dense MB (≥ 10x smaller) | 161.3 MB dense → 3.7 MB factors (**44x smaller**); 5.0 MB as a sparse matrix |
| 2 | The nightly job is fast | fit + neighbour table + top 10 for every customer, in seconds (< 60 s on a laptop) | **0.7 s** for fit + neighbour table + top 10 for all 5,471 customers |
| 3 | Recommendations beat best sellers | hit rate at 10 on held-out new purchases, model vs best-seller baseline | **39.4% vs 27.0%** (±2.2 points, 1,913 test customers) |
| 4 | Customer map shows groups marketing can name | 3 to 6 segments, each named from its products and countries | **6 segments**: heart & wicker home decor 23%, kids' toys & games 18%, vintage tea-party kitchenware 16%, novelty metal signs 16%, party tableware 14%, jumbo bags & storage 13% |

## Data

UCI Online Retail II (Chen, 2012, CC BY 4.0): <https://archive.ics.uci.edu/dataset/502/online+retail+ii>.
Put `online_retail_II.xlsx` in `data/raw/`. The `data/` folder is git-ignored and never committed.

Rows loaded: **1,067,371** (525,461 + 541,910 over two sheets). After cleaning: 776,577. Matrix: 5,471 customers × 3,686 products, 97.9% empty.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_matrix.py                                   # DE1–DE5 (~1.5 min first time): rebuilds everything in data/processed/ from the raw file
jupytext --to ipynb analysis.py                          # creates analysis.ipynb from analysis.py
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb   # ML2–ML7, runs top to bottom
```

Outputs: `data/processed/` (matrix, logs, report; git-ignored), `reports/` (committed copies of the profile, cleaning log and matrix report), `results/` (tables) and `figures/` (k curve, customer map).
All random components use seed 42.

## Files

| File | What it does | Tasks |
|------|--------------|-------|
| `build_matrix.py` | load → profile → clean → time split → sparse matrix → report | DE1–DE6 |
| `recsys.py` | fit SVD/PCA, errors, recommendations, hit rate, timing, segments | ML2–ML7 |
| `analysis.py` / `.ipynb` | the analysis notebook, top to bottom | ML1–ML7 |
| `DECISIONS.md` | every decision and its reason | R1 |
| `CONTRIBUTION_LOG.md` | task, owner, date, evidence | C2 |
