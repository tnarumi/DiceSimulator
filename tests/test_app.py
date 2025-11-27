import io

import numpy as np
import pytest

import app


def test_roll_sample_means_shape_and_seed_reproducible():
    means1 = app.roll_sample_means(sample_size=5, sample_count=4, seed=123)
    means2 = app.roll_sample_means(sample_size=5, sample_count=4, seed=123)
    assert means1.shape == (4,)
    np.testing.assert_allclose(means1, means2)
    assert np.all((means1 >= 1) & (means1 <= 6))


def test_cumulative_means_basic():
    data = np.array([1, 3, 5, 7], dtype=float)
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(app.cumulative_means(data), expected)


def test_plot_histogram_returns_image_bytes():
    means = np.array([3.0, 3.5, 4.0, 3.2, 3.8])
    buf = app.plot_histogram(means, sample_size=5)
    assert isinstance(buf, io.BytesIO)
    assert buf.getbuffer().nbytes > 0


def test_plot_cumulative_mean_sets_ylim(monkeypatch):
    # Capture the axes to inspect ylim before figure is closed.
    fig, ax = app.plt.subplots(figsize=(7, 4))

    def fake_subplots(*args, **kwargs):
        return fig, ax

    monkeypatch.setattr(app.plt, "subplots", fake_subplots)

    buf = app.plot_cumulative_mean_of_rolls(np.array([1, 2, 3, 4, 5]))

    assert ax.get_ylim() == (1, 6)
    assert isinstance(buf, io.BytesIO)
    assert buf.getbuffer().nbytes > 0

    app.plt.close(fig)
