# Decision log (R1)

| # | Decision | Reason |
|---|----------|--------|
| D1 | Use both sheets (Dec 2009 – Dec 2011) | Two years of history give more repeat behaviour to learn from |
| D2 | Drop lines with no Customer ID | Can't attribute them to a customer row |
| D3 | Drop cancellations (Invoice starting with "C") | A cancelled order is not a purchase |
| D4 | Drop Quantity ≤ 0 and Price ≤ 0 | Returns, adjustments, free samples and write-offs are not sales |
| D5 | Keep only StockCodes matching 5 digits + optional letters | POST, M, D, DOT, BANK CHARGES, AMAZONFEE, gift vouchers and test codes are not recommendable products |
| D6 | Drop exact duplicate lines | Logging duplicates would double-count purchases |
| D7 | Test period = last 2 months (time split, not random) | Mimics reality: predict what customers buy *next* from what they bought *before* |
| D8 | Held-out items = products a customer bought in the test period but never before | Recommending things they already reorder is trivial; the box should surface something new |
| D9 | Product filter: bought by ≥ 10 distinct training customers, decided on training data only | Products with a handful of buyers have unreliable factors and add noise; deciding on training data avoids leakage (R2) |
| D10 | Cell value = 1/0 "ever bought" | Chosen on the validation split: 1/0 gave 40.9% hit rate vs 37.5% for log(1+qty) and 20.6% for raw counts at k=50. Raw counts let a few wholesalers ordering thousands of units dominate (ML7) |
| D11 | Main method = Truncated SVD (decided before fitting, see notebook ML1) | Works natively on sparse data, gives customer and product factors in one fit, and zero stays "not bought" |
| D12 | k = 50, chosen on a validation split inside the training period (last 2 training months): the smallest k within 1 point of the best validation hit rate (41.8% at k=100) | The test set is used only once; k=50 is half the size of k=100 for almost the same accuracy |
| D13 | Customer top 10 = their row of the reconstruction A_k = U·S·Vᵀ (already-bought items skipped). Product-page box = 10 nearest products by cosine similarity of V·S vectors | On validation, reconstruction scoring beat summing item-item cosine similarities (40.9% vs 18.2% at k=50). Cosine is still the natural fit for "similar to this one product" |
| D14 | Baseline = 10 best sellers by number of distinct training buyers, skipping items the customer already bought | Fair comparison: same "only new items" rule as the model |
| D15 | Segments: k-means on length-normalised first 10 customer factors; 3–6 groups chosen by silhouette | Normalising groups customers by *what* they buy, not *how much* |
| D16 | No customer IDs in results/ or figures/; the index map lives only in data/processed (git-ignored) | Privacy, R4 |
