import numpy as np
import pytest

from ann.metrics import recall_at_k, recall_per_query, true_distances


def line_data():
    """Base vectors on a line at x = 0..19; one query at x = 0.

    Distances are 0, 1, 4, 9, ... so the 10 nearest are ids 0..9.
    """
    base = np.arange(20, dtype=np.float32).reshape(-1, 1)
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(20).reshape(1, -1)
    return base, q, gt


def test_perfect():
    base, q, gt = line_data()
    assert recall_at_k(base, q, gt[:, :10], gt) == 1.0


def test_no_overlap():
    base, q, gt = line_data()
    assert recall_at_k(base, q, np.arange(10, 20).reshape(1, -1), gt) == 0.0


def test_partial_and_order_independent():
    base, q, gt = line_data()
    found = np.array([[9, 8, 7, 6, 5, 15, 16, 17, 18, 19]])
    assert recall_at_k(base, q, found, gt) == pytest.approx(0.5)


def test_tie_at_kth_neighbour_is_not_a_mistake():
    # Query at 0, base at -9 and +9 are tied for the 10th neighbour.
    base = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4, 9, -9, 20],
                    dtype=np.float32).reshape(-1, 1)
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])     # 9 is 10th
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])         # tied 10 instead
    assert recall_at_k(base, q, found, gt) == 1.0
    found_worse = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]])   # 20 is not tied
    assert recall_at_k(base, q, found_worse, gt) == pytest.approx(0.9)


def test_exact_duplicate_vectors_are_interchangeable():
    # Ids 9 and 10 are the same vector: either one is a correct 10th answer.
    base = np.arange(20, dtype=np.float32).reshape(-1, 1)
    base[10] = base[9]
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(20).reshape(1, -1)
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])
    assert recall_at_k(base, q, found, gt) == 1.0


def test_missing_results_count_as_misses():
    base, q, gt = line_data()
    found = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1]])
    assert recall_at_k(base, q, found, gt) == pytest.approx(0.5)


def test_repeated_id_counts_once():
    base, q, gt = line_data()
    found = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert recall_at_k(base, q, found, gt) == pytest.approx(0.1)


def test_mean_over_queries():
    base, _, _ = line_data()
    q = np.array([[0.0], [19.0]], dtype=np.float32)
    gt = np.array([np.arange(20), np.arange(19, -1, -1)])
    found = np.array([np.arange(10), np.arange(10)])   # right for q0, wrong for q1
    assert recall_at_k(base, q, found, gt) == pytest.approx(0.5)


def test_true_distances_ignore_index_reported_values():
    base, q, _ = line_data()
    d = true_distances(base, q, np.array([[3, -1]]))
    assert d[0, 0] == 9.0 and np.isinf(d[0, 1])


def test_recall_per_query_direct():
    ids = np.array([[1, 2, 3]])
    dists = np.array([[1.0, 2.0, 5.0]])
    assert recall_per_query(ids, dists, np.array([2.0]), k=3)[0] == pytest.approx(2 / 3)
