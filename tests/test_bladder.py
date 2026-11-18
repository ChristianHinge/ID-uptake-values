import numpy as np

from id_uptake_values.bladder import BLADDER_WINDOW, cut_bladder
from id_uptake_values.models import KNNActivity
from id_uptake_values.estimation import knn_model

x = np.linspace(0.75, -3, 1024)
inside = (x < max(BLADDER_WINDOW)) & (x > min(BLADDER_WINDOW))


def test_spike_is_capped_and_rest_untouched():
    y = np.ones_like(x) + 5 * np.exp(-((x + 1.12) / 0.04) ** 2)
    y_cut = cut_bladder(y, x)
    np.testing.assert_allclose(y_cut[inside], 1.0, atol=1e-3)  # spike tail leaks slightly into the edge band
    np.testing.assert_array_equal(y_cut[~inside], y[~inside])


def test_curve_without_spike_is_unchanged():
    y = np.linspace(2, 1, x.size)  # linear, so the baseline lies on the curve
    np.testing.assert_allclose(cut_bladder(y, x), y, rtol=1e-3)


def test_batch_matches_single_curves():
    X = np.load(knn_model / "activity" / "X.npy")[:5]
    np.testing.assert_array_equal(cut_bladder(X, x), np.stack([cut_bladder(r, x) for r in X]))


def test_reference_curves_renormalised_only_when_enabled():
    on, off = KNNActivity(40, remove_bladder=True), KNNActivity(40)
    on.load_weights(knn_model / "activity")
    off.load_weights(knn_model / "activity")
    np.testing.assert_allclose(on.X.sum(axis=1), 1.0)
    np.testing.assert_array_equal(off.X, np.load(knn_model / "activity" / "X.npy"))
