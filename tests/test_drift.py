import numpy as np
import pytest

from ann import drift


def small(alpha, seed=drift.SEED):
    return drift.make(alpha, n_old=1000, n_new=500, seed=seed)


def test_same_seed_same_data():
    a, b = small(1.0), small(1.0)
    assert np.array_equal(a.new, b.new) and np.array_equal(a.q_new, b.q_new)


def test_different_seed_different_direction():
    assert not np.allclose(small(1.0).new, small(1.0, seed=drift.SEED + 1).new)


def test_control_is_unshifted():
    c = small(0.0)
    assert np.array_equal(c.q_new, c.q_old)
    assert np.array_equal(c.base[1000:], small(1.0).base[1000:] - (small(1.0).new - c.new))


def test_shift_size_and_direction():
    c, s = small(0.0), small(2.0)
    offset = s.new - c.new
    assert np.allclose(offset, offset[0], atol=1e-3)            # same offset for all
    assert np.linalg.norm(offset[0]) == pytest.approx(2.0 * s.spread, rel=1e-4)
    assert np.allclose(s.q_new - s.q_old, offset[0], atol=1e-3)  # queries move too
    assert np.array_equal(s.old, c.old)                          # old data untouched


def test_base_order_old_then_new():
    s = small(1.0)
    assert s.base.shape == (1500, 128)
    assert np.array_equal(s.base[:1000], s.old) and np.array_equal(s.base[1000:], s.new)
