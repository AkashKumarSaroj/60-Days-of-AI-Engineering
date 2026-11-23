import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def run_statistical_checks_and_baseline() -> None:
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "plan_tier": rng.choice(["Low", "Medium", "High"], size=n),
        "region": rng.choice(["APAC", "EMEA", "NA"], size=n),
        "api_calls": rng.lognormal(mean=4.5, sigma=0.9, size=n).round(1),
        "latency_ms": rng.normal(loc=180, scale=35, size=n).round(1),
    })
    df["upgraded"] = ((df["api_calls"] > 95) & (df["plan_tier"] != "Low")).astype(int)

    # 1. Skewness diagnostic: Mean vs. Median
    mean_calls, median_calls = df["api_calls"].mean(), df["api_calls"].median()
    print(f"API Calls -> Mean: {mean_calls:.2f} vs Median: {median_calls:.2f} (Positive Skew)")

    # 2. Pearson (linear) vs. Spearman (monotonic rank)
    r_val, _ = stats.pearsonr(df["api_calls"], df["upgraded"])
    rho_val, _ = stats.spearmanr(df["api_calls"], df["upgraded"])
    print(f"Pearson r: {r_val:.3f} | Spearman rho: {rho_val:.3f}")

    # 3. Chi-Square categorical independence test
    contingency = pd.crosstab(df["plan_tier"], df["upgraded"])
    _, p_val, _, _ = stats.chi2_contingency(contingency)
    print(f"Chi-Square p-value (plan_tier vs upgraded): {p_val:.5f}\n")

    # 4. Leakage-free preprocessing & DecisionTree baseline (70/30 split)
    X = df[["plan_tier", "region", "api_calls", "latency_ms"]]
    y = df["upgraded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", OrdinalEncoder(categories=[["Low", "Medium", "High"]]), ["plan_tier"]),
            ("nominal", OneHotEncoder(drop="first", handle_unknown="ignore"), ["region"]),
            ("numeric", StandardScaler(), ["api_calls", "latency_ms"]),
        ]
    )

    clf = Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"DecisionTree Baseline Holdout Accuracy: {accuracy_score(y_test, preds):.4f}")


if __name__ == "__main__":
    run_statistical_checks_and_baseline()
