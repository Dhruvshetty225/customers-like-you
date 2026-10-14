"""
recsys.py  -  ML Engineer toolkit (ML2 to ML7). The notebook analysis.ipynb calls these functions.

Everything is deterministic: SEED is passed to every random component (rule R6).
"""
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import build_matrix as bm

SEED = 42
DATA = Path("data/processed")
TOP_N = 10

# Validated categorical palette (fixed order, never cycled), from the reference palette.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


# ----------------------------------------------------------------------------- loading
def load(value="log"):
    A = sp.load_npz(DATA / f"A_{value}.npz").tocsr()
    customers = pd.read_csv(DATA / "customers.csv")
    products = pd.read_csv(DATA / "products.csv", dtype={"StockCode": str})
    holdout = pd.read_csv(DATA / "test_holdout.csv")
    report = json.loads((DATA / "matrix_report.json").read_text())
    return A, customers, products, holdout, report


def validation_split(min_customers=None):
    """Inner time split INSIDE the training period: the last 2 training months play the role of the
    test set. k and the purchase value are chosen on this validation set, so the real test set is
    only touched once, for the final numbers (R2)."""
    if min_customers is None:   # use the same product filter as the main matrix
        min_customers = json.loads((DATA / "matrix_report.json").read_text())["min_customers_per_product"]
    train = pd.read_parquet(DATA / "train_lines.parquet")
    inner_train, val, cutoff = bm.time_split(train)
    kept, cust, prod = bm.build_index(inner_train, min_customers)
    hold, stats = bm.make_holdout(kept, val, cust, prod)
    mats = {v: bm.build_matrix(kept, cust, prod, v) for v in ["log", "binary", "count"]}
    return mats, prod, hold, {"val_cutoff": str(cutoff.date()), **stats}


# ----------------------------------------------------------------------------- fitting (ML2, ML5)
def _timed_fit(model, A):
    tracemalloc.start()
    t = time.perf_counter()
    model.fit(A)
    secs = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return model, secs, peak / 1e6


def fit_svd(A, k):
    return _timed_fit(TruncatedSVD(n_components=k, random_state=SEED), A)


def fit_pca(A, k):
    # arpack centres implicitly, so the sparse matrix is never densified (scikit-learn >= 1.4)
    return _timed_fit(PCA(n_components=k, svd_solver="arpack", random_state=SEED), A)


def relative_error(A, model):
    """||A - A_k||_F / ||A||_F, computed exactly without ever building the dense A_k.
    For an orthonormal projection, ||A - A_k||^2 = ||A||^2 - ||projection||^2."""
    total = float(A.multiply(A).sum())
    if isinstance(model, PCA):
        n = A.shape[0]
        centred_total = total - n * float(model.mean_ @ model.mean_)
        T = model.transform(A)
        resid = centred_total - float((T ** 2).sum())
    else:
        X = model.transform(A)
        resid = total - float((X ** 2).sum())
    return float(np.sqrt(max(resid, 0.0) / total))


def factors_mb(A, k):
    n, m = A.shape
    return (n + m) * k * 8 / 1e6


# ----------------------------------------------------------------------------- recommending (ML3)
def product_vectors(model):
    """One row per product: its factor vector (V * singular values), scaled to length 1 so that a dot
    product is the cosine similarity."""
    return normalize(model.components_.T * model.singular_values_)


def similar_products(P, j, n=TOP_N):
    s = P @ P[j]
    s[j] = -np.inf
    return np.argsort(-s)[:n]


def all_neighbours(P, n=TOP_N):
    """The table behind the product-page box: 10 most similar products for every product."""
    S = (P @ P.T).astype(np.float32)
    np.fill_diagonal(S, -np.inf)
    top = np.argpartition(-S, n, axis=1)[:, :n]
    order = np.take_along_axis(S, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def _top_n_excluding_bought(scores, bought_mask, n=TOP_N):
    scores = np.where(bought_mask, -np.inf, scores)
    top = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def recommend(A, model, customers=None, n=TOP_N, batch=1000):
    """Customer top 10 = the customer's row of the rank-k reconstruction A_k = U S V^T.
    Compression smooths each row towards what similar customers buy, so a high value on a product
    the customer has NOT bought means "people like you buy this". Already-bought items are skipped.
    (Chosen on the validation split over summed item-item cosine similarity, see DECISIONS D13.)"""
    rows = np.arange(A.shape[0]) if customers is None else np.asarray(customers)
    US = model.transform(A[rows]).astype(np.float32)        # (customers x k) = U S  (PCA: centred)
    Vt = model.components_.astype(np.float32)                # (k x products)
    offset = getattr(model, "mean_", None)                   # PCA adds the average customer back
    out = np.empty((len(rows), n), dtype=np.int64)
    for s in range(0, len(rows), batch):
        sl = slice(s, s + batch)
        scores = US[sl] @ Vt
        if offset is not None:
            scores = scores + offset.astype(np.float32)
        bought = A[rows[sl]].toarray() > 0
        out[sl] = _top_n_excluding_bought(scores, bought, n)
    return rows, out


def recommend_bestsellers(A, n_customers_per_product, customers=None, n=TOP_N):
    """Baseline (R3): the 10 best-selling products (by number of distinct buyers in training)
    that this customer has not bought yet."""
    rows = np.arange(A.shape[0]) if customers is None else np.asarray(customers)
    order = np.argsort(-np.asarray(n_customers_per_product), kind="stable")
    cand = order[: n + 300]
    bought = A[rows][:, cand].toarray() > 0
    out = np.empty((len(rows), n), dtype=np.int64)
    for i in range(len(rows)):
        out[i] = cand[~bought[i]][:n]
    return rows, out


# ----------------------------------------------------------------------------- evaluation (ML4)
def hit_rate(rows, recs, holdout):
    """Share of evaluation customers with at least one held-out (new) product in their top 10.
    Returns rate, number of customers and a 95% confidence half-width."""
    truth = holdout.groupby("customer_idx")["product_idx"].apply(set)
    pos = {c: i for i, c in enumerate(rows)}
    hits = [bool(truth[c] & set(recs[pos[c]])) for c in truth.index]
    p, n = float(np.mean(hits)), len(hits)
    return {"hit_rate": p, "n_customers": n, "ci95": 1.96 * np.sqrt(p * (1 - p) / n)}


def evaluate(A, model, holdout, n_customers_per_product=None):
    eval_rows = np.sort(holdout["customer_idx"].unique())
    rows, recs = recommend(A, model, eval_rows)
    res = {"model": hit_rate(rows, recs, holdout)}
    if n_customers_per_product is not None:
        rows_b, recs_b = recommend_bestsellers(A, n_customers_per_product, eval_rows)
        res["bestsellers"] = hit_rate(rows_b, recs_b, holdout)
    return res


def nightly_job(A, k, method="svd"):
    """Time the whole overnight job: fit + product neighbour table + top 10 for every customer."""
    t0 = time.perf_counter()
    model, fit_s, peak = (fit_svd if method == "svd" else fit_pca)(A, k)
    P = product_vectors(model)
    t1 = time.perf_counter()
    neigh = all_neighbours(P)
    t2 = time.perf_counter()
    recommend(A, model)
    t3 = time.perf_counter()
    return {"fit_s": fit_s, "product_neighbours_s": t2 - t1, "score_all_customers_s": t3 - t2,
            "total_s": t3 - t0, "peak_fit_MB": peak, "neighbour_table_shape": list(neigh.shape)}


# ----------------------------------------------------------------------------- component 1 (ML5)
def describe_component_1(A, model, products, n=6):
    v = model.components_[0]
    if v.sum() < 0:   # sign of a component is arbitrary; flip so most loadings are positive
        v = -v
    score = model.transform(A)[:, 0] * (1 if model.components_[0].sum() >= 0 else -1)
    volume = np.asarray(A.sum(axis=1)).ravel()
    desc = products["Description"].fillna(products["StockCode"]).to_numpy()
    return {
        "share_of_products_with_same_sign": float(max((v > 0).mean(), (v < 0).mean())),
        "corr_with_customer_volume": float(np.corrcoef(score, volume)[0, 1]),
        "top_positive": list(desc[np.argsort(-v)[:n]]),
        "top_negative": list(desc[np.argsort(v)[:n]]),
    }


# ----------------------------------------------------------------------------- customer map (ML6)
def customer_segments(model, A, n_dims=10, k_range=range(3, 7)):
    """Cluster on taste, not size: take the first n_dims customer factors and scale each customer's
    vector to length 1, so a big wholesaler and a small shop with the same mix land together.
    The number of groups (3 to 6) is picked by silhouette score. The 2-D picture is the first two
    principal components of that taste space."""
    X = model.transform(A)[:, :n_dims]
    Z = normalize(X)
    rng = np.random.RandomState(SEED)
    sample = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    scores = {}
    for c in k_range:
        lab = KMeans(n_clusters=c, n_init=10, random_state=SEED).fit_predict(Z)
        scores[c] = float(silhouette_score(Z[sample], lab[sample], random_state=SEED))
    best = max(scores, key=scores.get)
    km = KMeans(n_clusters=best, n_init=10, random_state=SEED).fit(Z)
    # relabel so segment 0 is the largest (stable colours across runs)
    sizes = np.bincount(km.labels_)
    remap = {old: new for new, old in enumerate(np.argsort(-sizes))}
    labels = np.array([remap[l] for l in km.labels_])
    xy = PCA(n_components=2, random_state=SEED).fit_transform(Z)
    return labels, xy, scores


def segment_profiles(labels, A, customers, products, n_products=5):
    """Describe each segment by size, countries and the products that are most characteristic of it
    (lift = share of the segment buying it / share of all customers buying it). No customer IDs."""
    B = (A > 0).astype(np.float32)
    overall = np.asarray(B.mean(axis=0)).ravel()
    desc = products["Description"].fillna(products["StockCode"]).to_numpy()
    rows = []
    for s in np.unique(labels):
        mask = labels == s
        share = np.asarray(B[mask].mean(axis=0)).ravel()
        lift = np.where(share >= 0.15, share / np.maximum(overall, 1e-9), 0)   # ignore rare items
        top = np.argsort(-lift)[:n_products]
        countries = customers.loc[mask, "Country"].value_counts(normalize=True).head(3)
        rows.append({
            "segment": int(s),
            "customers": int(mask.sum()),
            "pct_customers": round(100 * mask.mean(), 1),
            "median_distinct_products": float(np.median(np.asarray(B[mask].sum(axis=1)).ravel())),
            "top_countries": ", ".join(f"{c} {p:.0%}" for c, p in countries.items()),
            "characteristic_products": " | ".join(f"{desc[j]} (x{lift[j]:.1f})" for j in top),
        })
    return pd.DataFrame(rows)


def plot_customer_map(xy, labels, names, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for s in np.unique(labels):
        m = labels == s
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.6, color=PALETTE[s], linewidths=0,
                   label=f"{names.get(s, f'Segment {s + 1}')} ({m.mean():.0%})")
    _style(ax, "Customer map: groups by what they buy",
           "Taste dimension 1", "Taste dimension 2")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2.5,
              fontsize=9, labelcolor=INK)
    ax.set_xticks([]); ax.set_yticks([])   # axes are abstract; no numbers to misread
    fig.text(0.01, 0.01, "Each dot is one customer (anonymous). No customer IDs shown.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return fig


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
