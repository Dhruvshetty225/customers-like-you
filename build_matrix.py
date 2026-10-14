"""
build_matrix.py  -  Data Engineer pipeline (DE1 to DE6)

One command rebuilds everything from the raw Excel file (rule R6):

    python build_matrix.py                      # default: 1/0 "ever bought", products bought by >= 10 customers
    python build_matrix.py --value log          # log(1 + quantity)
    python build_matrix.py --value count        # raw quantity

Outputs (all in data/processed/, which is git-ignored):
    profile.csv          DE1  rows, dtypes, missing values per column
    cleaning_log.csv     DE2  rows removed by each rule
    train_lines.parquet  DE3  cleaned training invoice lines (before the test cut-off)
    test_holdout.csv     DE3  held-out (customer_idx, product_idx) pairs = NEW products bought in the test period
    A_log/binary/count.npz  DE4  sparse customer x product matrices (training data only); --value picks the main one
    customers.csv        DE4  row index -> Customer ID   (PRIVATE, never leaves the team, rule R4)
    products.csv         DE4  column index -> StockCode, Description, n_customers
    matrix_report.json   DE5  shape, non-zeros, sparsity, memory dense vs sparse vs factors
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path("data/raw/online_retail_II.xlsx")
OUT = Path("data/processed")
REPORTS = Path("reports")       # committed copies of the logs/reports
CACHE = OUT / "raw_cache.parquet"   # reading 1M Excel rows takes minutes; cache once

# A real product code is 5 digits, optionally followed by letters for variants (e.g. 85123A, 15056BL).
# Everything else (POST, M, D, DOT, C2, BANK CHARGES, AMAZONFEE, gift_0001_10, TEST001, ADJUST ...) is not a product.
PRODUCT_CODE = re.compile(r"^\d{5}[A-Z]{0,3}$")

TEST_MONTHS = 2          # DE3: the last 2 months are the test period
SEED = 42


# ----------------------------------------------------------------------------- DE1
def load_raw(path: Path = RAW) -> pd.DataFrame:
    if CACHE.exists() and CACHE.stat().st_mtime > path.stat().st_mtime:
        return pd.read_parquet(CACHE)
    print(f"Reading {path} (both sheets, this can take a few minutes the first time)...")
    sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "InvoiceNo": str,
                                                         "StockCode": str})
    for name, s in sheets.items():
        print(f"  sheet {name!r}: {len(s):,} rows")
    df = pd.concat(sheets.values(), ignore_index=True)
    # Column names differ between versions of the file: normalise them.
    df = df.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price",
                            "CustomerID": "Customer ID"})
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    p.loc["__TOTAL_ROWS__", "non_null"] = len(df)
    return p


# ----------------------------------------------------------------------------- DE2
def clean(df: pd.DataFrame):
    """Apply each rule in order and log how many rows it removed (R1: every rule has a reason)."""
    rules = [
        ("missing Customer ID",
         "cannot attribute the purchase to a customer, so it cannot enter a customer x product matrix",
         lambda d: d["Customer ID"].notna()),
        ("cancellation (Invoice starts with 'C')",
         "a cancelled order is not a purchase; counting it would invent demand",
         lambda d: ~d["Invoice"].str.startswith("C")),
        ("Quantity <= 0",
         "returns and stock adjustments are not purchases",
         lambda d: d["Quantity"] > 0),
        ("Price <= 0",
         "free samples, write-offs and bad-debt adjustments are not real sales",
         lambda d: d["Price"] > 0),
        ("non-product StockCode (POST, M, D, BANK CHARGES, ...)",
         "postage, manual entries, fees and test codes are not products we could recommend",
         lambda d: d["StockCode"].str.match(PRODUCT_CODE)),
        ("exact duplicate line",
         "identical lines (same invoice, product, qty, time, price) are logging duplicates",
         lambda d: ~d.duplicated()),
    ]
    log = [{"rule": "raw rows", "rows_removed": 0, "rows_left": len(df), "reason": ""}]
    for name, reason, keep in rules:
        before = len(df)
        df = df[keep(df)]
        log.append({"rule": name, "rows_removed": before - len(df), "rows_left": len(df),
                    "reason": reason})
    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df, pd.DataFrame(log)


# ----------------------------------------------------------------------------- DE3
def time_split(df: pd.DataFrame):
    cutoff = df["InvoiceDate"].max().normalize() - pd.DateOffset(months=TEST_MONTHS)
    train = df[df["InvoiceDate"] < cutoff]
    test = df[df["InvoiceDate"] >= cutoff]
    return train, test, cutoff


# ----------------------------------------------------------------------------- DE4
def purchase_value(qty: pd.Series, how: str) -> np.ndarray:
    if how == "log":
        return np.log1p(qty.to_numpy(dtype=float))
    if how == "binary":
        return np.ones(len(qty))
    if how == "count":
        return qty.to_numpy(dtype=float)
    raise ValueError(how)


def build_index(train: pd.DataFrame, min_customers: int):
    """Product filter and index maps are decided on TRAINING data only (R2)."""
    n_cust_per_prod = train.groupby("StockCode")["Customer ID"].nunique()
    keep = n_cust_per_prod[n_cust_per_prod >= min_customers].index
    train_kept = train[train["StockCode"].isin(keep)]

    customers = np.sort(train_kept["Customer ID"].unique())
    products = np.sort(keep.to_numpy())
    # Most common description for each code (the same code can have slightly different names)
    desc = (train_kept.groupby("StockCode")["Description"]
            .agg(lambda s: s.mode().iloc[0] if s.notna().any() else ""))
    prod_df = pd.DataFrame({"product_idx": np.arange(len(products)), "StockCode": products,
                            "Description": desc.reindex(products).to_numpy(),
                            "n_customers": n_cust_per_prod.reindex(products).to_numpy()})
    cust_df = pd.DataFrame({"customer_idx": np.arange(len(customers)), "Customer ID": customers})
    country = train_kept.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iloc[0])
    cust_df["Country"] = country.reindex(customers).to_numpy()
    return train_kept, cust_df, prod_df


def build_matrix(train_kept, cust_df, prod_df, value="log"):
    agg = train_kept.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()
    rows = agg["Customer ID"].map(dict(zip(cust_df["Customer ID"], cust_df["customer_idx"])))
    cols = agg["StockCode"].map(dict(zip(prod_df["StockCode"], prod_df["product_idx"])))
    vals = purchase_value(agg["Quantity"], value)
    return sp.csr_matrix((vals, (rows.to_numpy(), cols.to_numpy())),
                         shape=(len(cust_df), len(prod_df)))


def make_holdout(train_kept, test, cust_df, prod_df):
    """For each customer we can recommend to (in the training matrix), hold out products they bought
    in the test period but NEVER bought before. Products outside the matrix cannot be recommended,
    so they are dropped and counted."""
    c_map = dict(zip(cust_df["Customer ID"], cust_df["customer_idx"]))
    p_map = dict(zip(prod_df["StockCode"], prod_df["product_idx"]))
    seen = set(zip(train_kept["Customer ID"], train_kept["StockCode"]))
    pairs = test[["Customer ID", "StockCode"]].drop_duplicates()
    stats = {"test_customer_product_pairs": len(pairs)}
    pairs = pairs[pairs["Customer ID"].isin(c_map)]
    stats["pairs_with_known_customer"] = len(pairs)
    is_new = [(c, p) not in seen for c, p in zip(pairs["Customer ID"], pairs["StockCode"])]
    pairs = pairs[is_new]
    stats["pairs_new_to_customer"] = len(pairs)
    pairs = pairs[pairs["StockCode"].isin(p_map)]
    stats["pairs_new_and_in_catalogue"] = len(pairs)
    hold = pd.DataFrame({"customer_idx": pairs["Customer ID"].map(c_map).to_numpy(),
                         "product_idx": pairs["StockCode"].map(p_map).to_numpy()})
    stats["eval_customers"] = int(hold["customer_idx"].nunique())
    return hold.sort_values(["customer_idx", "product_idx"]), stats


# ----------------------------------------------------------------------------- DE5
def matrix_report(A, k=50):
    n, m = A.shape
    sparse_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    return {
        "shape": [n, m],
        "non_zeros": int(A.nnz),
        "sparsity_pct": round(100 * (1 - A.nnz / (n * m)), 3),
        "avg_products_per_customer": round(A.nnz / n, 1),
        "dense_MB_float64": round(n * m * 8 / 1e6, 1),
        "sparse_MB_csr": round(sparse_bytes / 1e6, 2),
        f"factors_MB_k{k}": round((n + m) * k * 8 / 1e6, 2),
    }


# ----------------------------------------------------------------------------- main (DE6)
def main(value="binary", min_customers=10, raw=RAW):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    df = load_raw(Path(raw))
    profile(df).to_csv(OUT / "profile.csv")
    print(f"DE1  loaded {len(df):,} rows, columns {df.columns.tolist()}")

    clean_df, log = clean(df)
    log.to_csv(OUT / "cleaning_log.csv", index=False)
    print("DE2  cleaning log\n" + log[["rule", "rows_removed", "rows_left"]].to_string(index=False))

    train, test, cutoff = time_split(clean_df)
    train_kept, cust_df, prod_df = build_index(train, min_customers)
    hold, hstats = make_holdout(train_kept, test, cust_df, prod_df)
    train.to_parquet(OUT / "train_lines.parquet", index=False)
    hold.to_csv(OUT / "test_holdout.csv", index=False)
    print(f"DE3  cutoff {cutoff.date()} | train lines {len(train):,} | test lines {len(test):,}")
    print("     holdout:", hstats)

    # Build all three purchase-value versions (same rows/columns); `value` is the main one.
    # The other two are only used for the ML7 sensitivity check.
    for v in ["log", "binary", "count"]:
        sp.save_npz(OUT / f"A_{v}.npz", build_matrix(train_kept, cust_df, prod_df, v))
    A = sp.load_npz(OUT / f"A_{value}.npz")
    cust_df.to_csv(OUT / "customers.csv", index=False)
    prod_df.to_csv(OUT / "products.csv", index=False)

    report = matrix_report(A)
    report.update({"purchase_value": value, "min_customers_per_product": min_customers,
                   "test_cutoff": str(cutoff.date()), "test_months": TEST_MONTHS,
                   "products_dropped_by_filter": int(train["StockCode"].nunique() - len(prod_df)),
                   **hstats, "build_seconds": round(time.time() - t0, 1)})
    (OUT / "matrix_report.json").write_text(json.dumps(report, indent=2))
    # Copies that ARE committed to git (no customer IDs in any of them): DE1, DE2, DE5 evidence
    REPORTS.mkdir(exist_ok=True)
    for f in ["profile.csv", "cleaning_log.csv", "matrix_report.json"]:
        (REPORTS / f).write_bytes((OUT / f).read_bytes())
    print("DE5  matrix report\n" + json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--value", choices=["log", "binary", "count"], default="binary",
                    help="what one cell means (default 1/0 'ever bought'; chosen on validation, see DECISIONS D10)")
    ap.add_argument("--min-customers", type=int, default=10,
                    help="keep products bought by at least this many training customers")
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args()
    main(args.value, args.min_customers, args.raw)
