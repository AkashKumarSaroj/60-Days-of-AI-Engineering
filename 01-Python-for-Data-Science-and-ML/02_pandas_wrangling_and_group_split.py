import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def build_transaction_logs() -> pd.DataFrame:
    """Creates multi-row customer transaction data containing outliers, duplicates, and nulls."""
    return pd.DataFrame({
        "transaction_id": ["T01", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"],
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
        "category": ["Compute", "Compute", "Storage", "Compute", "Storage", "Compute", "API", "API", "Storage"],
        "spend_usd": [240.0, 240.0, 260.0, np.nan, 890.0, 210.0, np.nan, 230.0, 275.0],
        "units": [4, 4, 5, 3, 10, 4, 6, 4, 5],
        "churned": [0, 0, 0, 1, 1, 0, 0, 1, 1],
    })


def wrangle_and_audit_leakage() -> None:
    df = build_transaction_logs()

    # 1. Primary-key deduplication
    df = df.drop_duplicates(subset=["transaction_id"], keep="first").copy()

    # 2. True unit-economics aggregation (avoiding average-of-averages fallacy)
    cat_summary = (
        df.dropna(subset=["spend_usd"])
        .groupby("category", as_index=False)
        .agg(total_rev=("spend_usd", "sum"), total_units=("units", "sum"))
    )
    cat_summary["true_unit_price"] = (cat_summary["total_rev"] / cat_summary["total_units"]).round(2)
    print("=== Category Unit Economics ===")
    print(cat_summary.to_string(index=False), "\n")

    # 3. Random Split vs. Entity-Aware GroupShuffleSplit
    X = df[["customer_id", "spend_usd", "units"]]
    y = df["churned"]

    X_tr_rand, X_te_rand, _, _ = train_test_split(X, y, test_size=0.35, random_state=42)
    rand_overlap = set(X_tr_rand["customer_id"]).intersection(set(X_te_rand["customer_id"]))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df["customer_id"]))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    group_overlap = set(X_train["customer_id"]).intersection(set(X_test["customer_id"]))

    print(f"Random Split Customer Overlap (LEAKAGE): {rand_overlap}")
    print(f"Group Split Customer Overlap  (CLEAN)  : {group_overlap}\n")

    # 4. Leakage-free median imputation (fit strictly on X_train)
    imputer = SimpleImputer(strategy="median")
    X_train["spend_usd"] = imputer.fit_transform(X_train[["spend_usd"]])
    X_test["spend_usd"] = imputer.transform(X_test[["spend_usd"]])
    print("=== Leakage-Free Imputed Test Set ===")
    print(X_test.to_string(index=False))


if __name__ == "__main__":
    wrangle_and_audit_leakage()
