import numpy as np
import pytest
from ecgclf.preprocess import (
    segment,
    segment_beats,
    zscore,
    SEGMENT_SAMPLES,
    compute_class_weights,
    stratified_kfold_indices,
    train_val_test_split,
)


def test_segment_shape_and_nan_free():
    sig = np.sin(np.linspace(0, 100, 2000, dtype=np.float32))
    out = segment(sig, 5)  # near start, will pad
    assert out.shape == (SEGMENT_SAMPLES,)
    assert np.isfinite(out).all()


def test_segment_beats_length_and_labels():
    sig = np.random.randn(3000).astype(np.float32)
    r_locs = [100, 500, 2500]
    labels = [0, 1, 0]
    X, y = segment_beats(sig, r_locs, labels, pre=500, post=500, apply_z=True)
    assert X.shape == (3, 1000)
    assert y.tolist() == labels
    assert np.isfinite(X).all()


def test_zscore_normalization():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    z = zscore(x)
    # Mean approx 0, std approx 1
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std()) - 1.0) < 1e-6


def test_class_weights_balanced():
    y = np.array([0] * 90 + [1] * 10, dtype=int)
    cw = compute_class_weights(y)
    assert cw[1] > cw[0]  # minority class gets higher weight


def test_stratified_splits_basic():
    pytest.importorskip("sklearn")
    # balanced classes
    y = np.array([0, 1] * 50, dtype=int)
    folds = list(stratified_kfold_indices(y, n_splits=5, random_state=0))
    assert len(folds) == 5
    for tr, va in folds:
        # each fold should contain both classes
        assert set(y[va]) == {0, 1}


def test_train_val_test_split_shapes():
    pytest.importorskip("sklearn")
    X = np.random.randn(100, 1000).astype(np.float32)
    y = np.array([0, 1] * 50, dtype=int)
    (Xt, yt), (Xv, yv), (Xs, ys) = train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=0)
    assert Xt.shape[0] + Xv.shape[0] + Xs.shape[0] == 100
    assert yt.size + yv.size + ys.size == 100
