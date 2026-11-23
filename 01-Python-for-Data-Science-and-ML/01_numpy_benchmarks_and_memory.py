import time
import numpy as np


def benchmark_vectorization(n: int = 1_000_000) -> None:
    """Compares standard Python list comprehension vs. NumPy SIMD vectorized operations."""
    py_list = list(range(n))
    np_arr = np.arange(n, dtype=np.float64)

    t0 = time.perf_counter()
    _ = [x * 2.5 + 10.0 for x in py_list]
    py_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = np_arr * 2.5 + 10.0
    np_time = time.perf_counter() - t1

    print(f"Python List Time : {py_time:.4f}s")
    print(f"NumPy Vector Time: {np_time:.4f}s (Speedup: {py_time / np_time:.1f}x)\n")


def demonstrate_view_vs_copy() -> None:
    """Demonstrates zero-copy memory sharing (.ravel) vs. independent memory copy (.flatten)."""
    matrix = np.array([[10, 20, 30], [40, 50, 60]])

    ravel_view = matrix.ravel()
    flatten_copy = matrix.flatten()

    ravel_view[0] = 999  # Mutates original buffer
    flatten_copy[1] = 888  # Leaves original buffer untouched

    print("Original Matrix after ravel_view[0]=999 & flatten_copy[1]=888:")
    print(matrix, "\n")


def axis_and_masking_ops() -> None:
    """Demonstrates axis collapse rules and vectorized conditional masking."""
    scores = np.array([[78, 92, 65], [88, 45, 95], [52, 81, 74]])

    col_means = np.mean(scores, axis=0)  # Collapses rows -> per-column metric
    row_means = np.mean(scores, axis=1)  # Collapses columns -> per-row metric
    status_flags = np.where(scores >= 75, "PASS", "REVIEW")

    print(f"Column Means (axis=0): {np.round(col_means, 2)}")
    print(f"Row Means    (axis=1): {np.round(row_means, 2)}")
    print("Vectorized Status Mask:\n", status_flags)


if __name__ == "__main__":
    benchmark_vectorization()
    demonstrate_view_vs_copy()
    axis_and_masking_ops()
