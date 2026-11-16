"""The parts of surgery.lab that need no model: the arithmetic of a removed direction and of a
low-rank update."""

import numpy as np

from surgery import lab


def test_unit_rows_are_orthonormal_and_span_the_input():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(3, 16))
    Q = lab.unit_rows(R)
    assert Q.shape == (3, 16)
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-5)
    assert np.allclose(R - (R @ Q.T) @ Q, 0, atol=1e-5)  # nothing of R lies outside the span of Q
    assert lab.unit_rows(R[0]).shape == (1, 16)


def test_removing_a_direction_from_an_output_is_a_rank_one_weight_edit():
    rng = np.random.default_rng(1)
    W, x = rng.normal(size=(16, 8)), rng.normal(size=(5, 8))
    Q = lab.unit_rows(rng.normal(size=16)).astype(np.float64)
    y = x @ W.T
    removed = y - (y @ Q.T) @ Q  # what Tap.ablate does to a writer's output
    edited = x @ ((np.eye(16) - Q.T @ Q) @ W).T  # the weight edit it stands for
    assert np.allclose(removed, edited)
    assert np.allclose(removed @ Q.T, 0, atol=1e-5)


def test_shared_update_is_b_transpose_a_per_layer():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(2, 3, 6)).astype(np.float32), rng.normal(size=(2, 3, 10)).astype(np.float32)
    upd = lab.Tap.shared_update([4, 9], A, B)
    assert list(upd) == ["layers.4.mlp.shared_expert.down_proj.weight", "layers.9.mlp.shared_expert.down_proj.weight"]
    hid = rng.normal(size=(7, 6)).astype(np.float32)
    assert np.allclose((hid @ A[1].T) @ B[1], hid @ upd["layers.9.mlp.shared_expert.down_proj.weight"].T, atol=1e-4)
