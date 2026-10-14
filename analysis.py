# %% [markdown]
# # Customers Like You: ML analysis (ML1 to ML7)
#
# Run `python build_matrix.py` first. Then run this notebook top to bottom (Kernel > Restart & Run All).
# All results land in `results/` and all figures in `figures/`. No customer IDs appear in either (R4).

# %%
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import recsys as rs

np.random.seed(rs.SEED)
Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
pd.set_option("display.max_colwidth", 120)

VALUE = "binary"   # cell value; chosen on the validation split (see ML7 and DECISIONS D10)
A, customers, products, holdout, report = rs.load(VALUE)
print(json.dumps(report, indent=2))

# %% [markdown]
# ## ML1: Method decision (written before running anything)
#
# **Main model: Truncated SVD.**
#
# 1. The matrix is over 98% zeros. Truncated SVD works directly on the sparse matrix. PCA must
#    subtract the average customer first, which turns every zero into a small negative number, so
#    the data stops being sparse. Modern solvers centre implicitly, but they still pay for it in
#    time and memory.
# 2. One fit gives us both customer factors (U·S) and product factors (V). Product factors are
#    exactly what the "also bought" box needs.
# 3. PCA describes *how a customer differs from the average customer*. For recommendations we care
#    about *what a customer actually buys*, and zeros should mean "didn't buy", not
#    "bought less than average".
#
# **What would change my mind:** if PCA gave a clearly higher validation hit rate at the same k.
# ML5 checks this.

# %% [markdown]
# ## ML2: How much do we lose at each k? Which k do we keep?
#
# * **Left plot:** reconstruction error on the full training matrix (what compression throws away).
# * **Right plot:** hit rate on a *validation* split, i.e. the last 2 months *of the training
#   period*. We pick k here, so the real test set is used only once, at the end (R2).
#
# **Rule for choosing k:** the smallest k whose validation hit rate is within 1 percentage point
# of the best one. Smaller k means a smaller table and a faster job.

# %%
val_mats, val_products, val_hold, val_stats = rs.validation_split()
A_val = val_mats[VALUE]
print("validation set:", val_stats)
# k must be smaller than the number of products (only matters on tiny test data)
KS = [k for k in [10, 20, 50, 100, 200] if k < min(A.shape + A_val.shape) - 1]

rows = []
for k in KS:
    svd, svd_s, svd_mb = rs.fit_svd(A, k)
    pca, pca_s, pca_mb = rs.fit_pca(A, k)
    svd_val, _, _ = rs.fit_svd(A_val, k)
    hr = rs.evaluate(A_val, svd_val, val_hold)["model"]
    rows.append({"k": k,
                 "svd_rel_error": rs.relative_error(A, svd),
                 "svd_explained_var": float(svd.explained_variance_ratio_.sum()),
                 "pca_rel_error": rs.relative_error(A, pca),
                 "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
                 "svd_fit_s": svd_s, "pca_fit_s": pca_s,
                 "factors_MB": rs.factors_mb(A, k),
                 "val_hit_rate_svd": hr["hit_rate"]})
    print(f"k={k:>3}  done")
sweep = pd.DataFrame(rows)
_, val_base = rs.recommend_bestsellers(A_val, val_products["n_customers"],
                                       np.sort(val_hold["customer_idx"].unique()))
val_baseline = rs.hit_rate(np.sort(val_hold["customer_idx"].unique()), val_base, val_hold)["hit_rate"]

best = sweep["val_hit_rate_svd"].max()
K = int(sweep.loc[sweep["val_hit_rate_svd"] >= best - 0.01, "k"].min())
sweep.to_csv("results/k_sweep.csv", index=False)
print(f"\nChosen k = {K}  (validation best {best:.1%}, best-seller baseline {val_baseline:.1%})")
sweep.round(4)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
ax1.plot(sweep.k, sweep.svd_rel_error, marker="o", ms=5, lw=2, color=rs.PALETTE[0], label="Truncated SVD")
ax1.plot(sweep.k, sweep.pca_rel_error, marker="s", ms=5, lw=2, color=rs.PALETTE[1], label="PCA (centred)")
rs._style(ax1, "What compression loses", "k (number of components)",
          "Relative reconstruction error ||A − A_k|| / ||A||")
ax1.set_xscale("log"); ax1.set_xticks(KS); ax1.set_xticklabels(KS)
ax1.legend(frameon=False, fontsize=9)
ax1.axvline(K, color=rs.MUTED, lw=1, ls="--")

ax2.plot(sweep.k, sweep.val_hit_rate_svd * 100, marker="o", ms=5, lw=2, color=rs.PALETTE[0], label="SVD recommender")
ax2.axhline(val_baseline * 100, color=rs.MUTED, lw=1.5, ls=":", label="Best-seller baseline")
rs._style(ax2, "What the recommender gains (validation)", "k (number of components)", "Hit rate at 10 (%)")
ax2.set_xscale("log"); ax2.set_xticks(KS); ax2.set_xticklabels(KS)
ax2.axvline(K, color=rs.MUTED, lw=1, ls="--")
ax2.annotate(f"chosen k = {K}", (K, ax2.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
             fontsize=8, color=rs.MUTED)
ax2.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/k_curve.png", bbox_inches="tight")

# %% [markdown]
# ## ML3: Recommendations
#
# * **Product page box:** the 10 products whose factor vectors have the highest cosine similarity.
# * **For a customer:** take their row of the compressed matrix, A_k = U·S·Vᵀ. Compression fills in
#   the blanks with "what customers like you buy". We show the 10 highest products they haven't
#   bought yet. On validation this beat summing cosine similarities (DECISIONS D13).
#
# The sanity check below shows the neighbours of the 5 most popular products. They should look
# like things a shop would plausibly buy together.

# %%
svd, _, _ = rs.fit_svd(A, K)
P = rs.product_vectors(svd)
desc = products["Description"].fillna(products["StockCode"]).to_numpy()
examples = []
for j in np.argsort(-products["n_customers"].to_numpy())[:5]:
    for rank, nb in enumerate(rs.similar_products(P, j)[:5], 1):
        examples.append({"product": desc[j], "rank": rank, "also_bought": desc[nb]})
examples = pd.DataFrame(examples)
examples.to_csv("results/example_neighbours.csv", index=False)
examples

# %% [markdown]
# ## ML4: Final evaluation on held-out test purchases (touched once)
#
# **Hit rate at 10:** the share of test customers for whom at least one product they bought *for
# the first time* in the last 2 months appears in our top 10. The **baseline** gives every
# customer the 10 best sellers they don't already own.

# %%
test_res = rs.evaluate(A, svd, holdout, products["n_customers"])
job = rs.nightly_job(A, K, "svd")
results = pd.DataFrame({
    "approach": [f"Truncated SVD, k={K}", "Best sellers (baseline)"],
    "hit_rate_at_10": [test_res["model"]["hit_rate"], test_res["bestsellers"]["hit_rate"]],
    "ci95_plus_minus": [test_res["model"]["ci95"], test_res["bestsellers"]["ci95"]],
    "test_customers": [test_res["model"]["n_customers"]] * 2,
})
results.to_csv("results/test_results.csv", index=False)
print("Nightly job:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in job.items()})
results

# %% [markdown]
# ## ML5: PCA vs Truncated SVD, same matrix, same k

# %%
pca, pca_s, pca_mb = rs.fit_pca(A, K)
svd_fit, svd_s, svd_mb = rs.fit_svd(A, K)
pca_test = rs.evaluate(A, pca, holdout)["model"]
c1_svd = rs.describe_component_1(A, svd, products)
c1_pca = rs.describe_component_1(A, pca, products)

def c1_words(c):
    return (f"{c['share_of_products_with_same_sign']:.0%} of products load the same way; "
            f"corr. with customer volume {c['corr_with_customer_volume']:.2f}; "
            f"top: {', '.join(c['top_positive'][:3])}")

comparison = pd.DataFrame({
    "Measure": ["k used", "Fit time (seconds)", "Peak memory during fit (MB)", "Size of stored factors (MB)",
                "Relative reconstruction error", "Variance explained", "Hit rate at 10 on held-out purchases",
                "Best-seller baseline hit rate at 10", "What component 1 represents (evidence)"],
    "PCA": [K, round(pca_s, 2), round(pca_mb, 1), round(rs.factors_mb(A, K), 2),
            round(rs.relative_error(A, pca), 4), round(float(pca.explained_variance_ratio_.sum()), 4),
            f"{pca_test['hit_rate']:.1%}", f"{test_res['bestsellers']['hit_rate']:.1%}", c1_words(c1_pca)],
    "Truncated SVD": [K, round(svd_s, 2), round(svd_mb, 1), round(rs.factors_mb(A, K), 2),
                      round(rs.relative_error(A, svd_fit), 4), round(float(svd_fit.explained_variance_ratio_.sum()), 4),
                      f"{test_res['model']['hit_rate']:.1%}", f"{test_res['bestsellers']['hit_rate']:.1%}",
                      c1_words(c1_svd)],
})
comparison.to_csv("results/pca_vs_svd.csv", index=False)
json.dump({"svd": c1_svd, "pca": c1_pca}, open("results/component_1.json", "w"), indent=2)
comparison

# %% [markdown]
# **How to read component 1.** If almost all products load the same way and the score correlates
# strongly with customer volume, SVD's component 1 means *"how much this customer buys overall"*.
# PCA removes the average first, so its component 1 is a *contrast*: one group of products
# against another. Write the final sentence for the memo from the numbers above.

# %% [markdown]
# ## ML6: Customer map for marketing
#
# We cluster on *taste*, not size: each customer's factor vector is scaled to length 1. Silhouette
# score picks the number of groups (3 to 6). After the first run, read the profile table and put
# real names in `SEGMENT_NAMES`, then re-run this cell.

# %%
labels, xy, sil = rs.customer_segments(svd, A)
print("silhouette by number of groups:", {k: round(v, 3) for k, v in sil.items()})
profiles = rs.segment_profiles(labels, A, customers, products)
profiles.to_csv("results/segments.csv", index=False)
profiles

# %%
# Fill these in after reading the table above (segment number -> name marketing will use)
SEGMENT_NAMES = {
    0: "Heart & wicker home decor",
    1: "Kids' toys & traditional games",
    2: "Vintage tea-party kitchenware",
    3: "Novelty metal-sign shops",
    4: "Party tableware (most international)",
    5: "Jumbo bags & storage (biggest baskets)",
}
rs.plot_customer_map(xy, labels, SEGMENT_NAMES, "figures/customer_map.png");

# %% [markdown]
# ## ML7: Sensitivity. Does the purchase value matter?
#
# Same k and same validation split. The purchase value was chosen on **validation** (1/0 won);
# test numbers are shown for information only.

# %%
sens = []
for v in ["log", "binary", "count"]:
    m_val, _, _ = rs.fit_svd(val_mats[v], K)
    A_v = rs.load(v)[0]
    m_test, _, _ = rs.fit_svd(A_v, K)
    sens.append({"purchase_value": v,
                 "val_hit_rate": rs.evaluate(val_mats[v], m_val, val_hold)["model"]["hit_rate"],
                 "test_hit_rate": rs.evaluate(A_v, m_test, holdout)["model"]["hit_rate"]})
sens = pd.DataFrame(sens)
sens.to_csv("results/sensitivity.csv", index=False)
sens

# %% [markdown]
# ## Headline numbers for the memo

# %%
summary = {
    "k": K,
    "matrix_shape": report["shape"],
    "sparsity_pct": report["sparsity_pct"],
    "dense_MB": report["dense_MB_float64"],
    "sparse_MB": report["sparse_MB_csr"],
    "factors_MB": round(rs.factors_mb(A, K), 2),
    "shrink_factor_vs_dense": round(report["dense_MB_float64"] / rs.factors_mb(A, K), 1),
    "relative_error_svd": round(rs.relative_error(A, svd), 4),
    "hit_rate_svd": round(test_res["model"]["hit_rate"], 4),
    "hit_rate_pca": round(pca_test["hit_rate"], 4),
    "hit_rate_bestsellers": round(test_res["bestsellers"]["hit_rate"], 4),
    "test_customers": test_res["model"]["n_customers"],
    "nightly_job_seconds": round(job["total_s"], 2),
    "n_segments": int(len(np.unique(labels))),
}
json.dump(summary, open("results/summary.json", "w"), indent=2)
summary
