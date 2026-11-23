# Module 01: Python for Data Science & Machine Learning Essential Training (Part 1)

**Roadmap Coverage:** Days 1 – 12 (`#60DaysOfAIEngineering`)  
**Core Stack:** Python, NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-Learn, BeautifulSoup, Streamlit

---

## Engineering Milestones Covered (Days 1–12)

| Days | Topic | Key Engineering Takeaways |
| :--- | :--- | :--- |
| **Day 1–3** | **NumPy Memory & Vectorization** | Contiguous SIMD execution (~27x speedup over Python loops), zero-copy `.ravel()` views vs. `.flatten()` copies, axis collapses (`axis=0/1`), and `np.where` masking. |
| **Day 4–5** | **Pandas Wrangling & Group Leakage** | Outlier-resistant median imputation, primary-key deduplication, Split-Apply-Combine aggregations, and `GroupShuffleSplit` to prevent entity-level train-test leakage. |
| **Day 6–8** | **Visualization Architecture** | Object-oriented Matplotlib (`fig, ax`), SLA threshold callouts (`ax.axhline`), time-series resampling, and Seaborn `1.5 * IQR` / bootstrap regression diagnostics. |
| **Day 9–10** | **Statistical EDA & Baseline ML** | Skewness checks (`mean` vs. `median`), Pearson ($r$) vs. Spearman ($\rho$), $\chi^2$ independence tests, leakage-free train-only scaling, and `DecisionTreeClassifier` floor benchmarking. |
| **Day 11–12** | **Ingestion & Stakeholder Apps** | Defensive HTTP scraping (`User-Agent`, `raise_for_status()`, `re.compile` in BeautifulSoup) connected to reactive Python dashboards in Streamlit. |

---

## Scripts in This Module

1. `01_numpy_benchmarks_and_memory.py` — Vectorization timing benchmark, memory view vs. copy verification, and multidimensional axis reductions.
2. `02_pandas_wrangling_and_group_split.py` — Median imputation, unit-economics `groupby().agg()`, and `GroupShuffleSplit` leakage audit.
3. `03_statistical_eda_and_baseline_ml.py` — Skewness & $\chi^2$ statistical checks, `ColumnTransformer` preprocessing, and baseline Decision Tree classifier.
4. `04_web_scraping_and_streamlit_app.py` — Defensive BeautifulSoup HTML parser + Streamlit interactive analytics dashboard (`streamlit run 04_web_scraping_and_streamlit_app.py`).
