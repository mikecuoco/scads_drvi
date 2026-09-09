"""viz: colour policy, frugal boxes, and figures that touch no filesystem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scads_drvi.viz.color import (  # noqa: E402
    DEFAULT_RAMP,
    ROBUST_LIMITS,
    Z_NOMINAL_ONE_TAILED,
    SignificanceRamp,
    add_threshold_lines,
    categorical_palette,
    robust_limits,
    robust_norm,
    significance_class,
    significance_colors,
    significance_handles,
)
from scads_drvi.viz.frugal import (  # noqa: E402
    box_stats,
    box_stats_by_column,
    draw_boxes,
)
from scads_drvi.viz.save import SaveSpec, figure_metadata, save_figure  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestRobustLimits:
    def test_one_documented_default(self):
        assert ROBUST_LIMITS == (1.0, 99.5)

    def test_clips_the_tails(self):
        values = np.concatenate([np.zeros(1), np.arange(1, 100), [1e6]])
        low, high = robust_limits(values)
        assert high < 1e6

    def test_all_equal_is_widened_not_degenerate(self):
        """A zero-width normalisation renders as one flat colour and reads as no data."""
        low, high = robust_limits(np.full(10, 3.0))
        assert high > low

    def test_empty_and_all_nan_give_a_usable_range(self):
        assert robust_limits(np.array([])) == (0.0, 1.0)
        assert robust_limits(np.array([np.nan, np.nan])) == (0.0, 1.0)

    def test_nonzero_only_ignores_the_zero_spike(self):
        values = np.concatenate([np.zeros(900), np.linspace(1, 2, 100)])
        with_zeros = robust_limits(values)
        without = robust_limits(values, nonzero_only=True)
        assert without[0] > with_zeros[0]

    def test_symmetric_centres_on_zero(self):
        low, high = robust_limits(np.array([-1.0, 5.0]), symmetric=True)
        assert low == pytest.approx(-high)

    def test_floor_raises_the_lower_limit(self):
        low, _ = robust_limits(np.arange(100.0), floor=50.0)
        assert low == 50.0

    def test_bad_percentiles_are_refused(self):
        with pytest.raises(ValueError, match="0 <= low < high <= 100"):
            robust_limits(np.arange(10.0), percentiles=(90.0, 10.0))

    def test_norm_is_a_matplotlib_norm(self):
        norm = robust_norm(np.arange(100.0))
        assert norm(np.array([50.0])).shape == (1,)

    def test_gamma_gives_a_power_norm(self):
        from matplotlib.colors import PowerNorm

        assert isinstance(robust_norm(np.arange(1.0, 100.0), gamma=0.5), PowerNorm)

    def test_power_norm_refuses_negative_data(self):
        with pytest.raises(ValueError, match="vmin >= 0"):
            robust_norm(np.array([-1.0, 1.0]), gamma=0.5)


class TestSignificanceRamp:
    def test_thresholds_and_colours_stay_consistent(self):
        with pytest.raises(ValueError, match="need 3 colours"):
            SignificanceRamp(thresholds=(0.05, 0.10), colors=("a", "b"))

    def test_thresholds_must_ascend(self):
        with pytest.raises(ValueError, match="ascending"):
            SignificanceRamp(thresholds=(0.10, 0.05), colors=("a", "b", "c"))

    def test_classes_are_ordinal(self):
        got = significance_class(np.array([0.001, 0.07, 0.5]))
        assert list(got) == [0, 1, 2]

    def test_nan_is_least_significant(self):
        """A test that could not be evaluated is not evidence."""
        assert significance_class(np.array([np.nan]))[0] == 2

    def test_colours_come_from_the_ramp(self):
        colours = significance_colors(np.array([0.001, 0.5]))
        assert colours[0] == DEFAULT_RAMP.colors[0]
        assert colours[1] == DEFAULT_RAMP.colors[-1]

    def test_legend_labels_and_handles_agree_in_length(self):
        assert len(significance_handles()) == len(DEFAULT_RAMP.legend_labels)

    def test_nominal_z_is_derived_not_rounded(self):
        from scipy.stats import norm as normal

        assert Z_NOMINAL_ONE_TAILED == pytest.approx(normal.isf(0.05), abs=1e-12)
        assert Z_NOMINAL_ONE_TAILED != 1.645


class TestCategoricalPalette:
    def test_stable_across_subsets_of_the_same_full_list(self):
        """Two panels of one figure must not give a category two colours."""
        full = ["a", "b", "c", "d"]
        first = categorical_palette(full)
        second = categorical_palette(full)
        assert first == second

    def test_order_of_the_input_does_not_matter(self):
        assert categorical_palette(["b", "a"]) == categorical_palette(["a", "b"])

    def test_highlight_is_an_argument_not_a_baked_in_name(self):
        palette = categorical_palette(["x", "y"], highlight="y")
        assert palette["y"] == "#D55E00"
        assert palette["x"] != palette["y"]

    def test_unknown_highlight_is_refused(self):
        with pytest.raises(KeyError, match="cannot highlight"):
            categorical_palette(["x"], highlight="absent")

    def test_many_categories_all_get_distinct_colours(self):
        palette = categorical_palette([f"c{i}" for i in range(29)])
        assert len(set(palette.values())) == 29

    def test_empty_is_empty(self):
        assert categorical_palette([]) == {}


class TestThresholdLines:
    def test_draws_on_the_named_axis(self):
        fig, ax = plt.subplots()
        add_threshold_lines(ax, axis="y", bh=3.7)
        assert len(ax.lines) == 3

    def test_bad_axis_is_refused(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="'x' or 'y'"):
            add_threshold_lines(ax, axis="z")


class TestBoxStats:
    def test_whiskers_are_at_the_quartiles_by_default(self):
        """Unusual, inherited, and previously undocumented -- so it is named."""
        stats = box_stats(np.arange(100.0))
        assert stats.whis == "quartile"
        assert stats.whislo == pytest.approx(stats.q1)
        assert stats.whishi == pytest.approx(stats.q3)

    def test_iqr_whiskers_reach_further(self):
        stats = box_stats(np.arange(100.0), whis="1.5iqr")
        assert stats.whislo[0] <= stats.q1[0]
        assert stats.whishi[0] >= stats.q3[0]

    def test_groups_become_boxes(self):
        values = np.arange(30.0)
        groups = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
        stats = box_stats(values, groups)
        assert stats.labels == ("a", "b", "c")
        assert list(stats.n) == [10, 10, 10]

    def test_thin_groups_are_omitted(self):
        values = np.arange(12.0)
        groups = ["a"] * 10 + ["b"] * 2
        stats = box_stats(values, groups, min_n=5)
        assert stats.labels == ("a",)

    def test_no_group_reaches_min_n_is_an_error(self):
        with pytest.raises(ValueError, match="min_n"):
            box_stats(np.arange(3.0), ["a", "b", "c"], min_n=5)

    def test_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="same length"):
            box_stats(np.arange(3.0), ["a", "b"])

    def test_bxp_payload_has_the_keys_matplotlib_wants(self):
        payload = box_stats(np.arange(20.0), ["a"] * 20).to_bxp()
        assert set(payload[0]) == {
            "label", "med", "q1", "q3", "whislo", "whishi", "fliers"
        }

    def test_order_and_select(self):
        values = np.concatenate([np.zeros(10), np.ones(10) * 5])
        groups = ["low"] * 10 + ["high"] * 10
        stats = box_stats(values, groups)
        assert stats.order_by("median").labels[0] == "high"
        assert stats.select(["low"]).labels == ("low",)
        with pytest.raises(KeyError, match="no box for"):
            stats.select(["absent"])

    def test_by_column_uses_frame_columns(self):
        frame = pd.DataFrame(np.arange(20.0).reshape(5, 4), columns=list("abcd"))
        stats = box_stats_by_column(frame)
        assert stats.labels == ("a", "b", "c", "d")
        assert list(stats.n) == [5, 5, 5, 5]

    def test_by_column_label_count_is_checked(self):
        with pytest.raises(ValueError, match="labels for"):
            box_stats_by_column(np.zeros((3, 4)), ["only", "two"])

    def test_by_column_needs_a_matrix(self):
        with pytest.raises(ValueError, match="2-D"):
            box_stats_by_column(np.zeros(4))

    def test_by_column_matches_the_grouped_path(self):
        rng = np.random.default_rng(0)
        matrix = rng.random((200, 3))
        by_col = box_stats_by_column(matrix, ["x", "y", "z"])
        for slot, name in enumerate(("x", "y", "z")):
            one = box_stats(matrix[:, slot], [name] * 200)
            assert one.median[0] == pytest.approx(by_col.median[slot])

    def test_draw_boxes_puts_artists_on_the_axes(self):
        stats = box_stats(np.arange(40.0), ["a"] * 20 + ["b"] * 20, with_maximum=True)
        fig, ax = plt.subplots()
        draw_boxes(ax, stats, colors=["#111111", "#222222"], show_maximum=True, hline=0)
        assert len(ax.patches) == 2
        assert len(ax.lines) > 0


class TestSaveFigure:
    def test_writes_every_requested_format(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        written = save_figure(fig, "panel", tmp_path)
        assert [p.name for p in written] == ["panel.pdf", "panel.png"]
        assert all(p.exists() and p.stat().st_size > 0 for p in written)

    def test_creates_the_directory(self, tmp_path):
        fig, _ = plt.subplots()
        out = tmp_path / "deep" / "nested"
        save_figure(fig, "p", out)
        assert out.is_dir()

    def test_an_extension_in_the_name_is_refused(self, tmp_path):
        fig, _ = plt.subplots()
        with pytest.raises(ValueError, match="no extension"):
            save_figure(fig, "panel.pdf", tmp_path)

    def test_single_format_spec(self, tmp_path):
        fig, _ = plt.subplots()
        written = save_figure(
            fig, "p", tmp_path, spec=SaveSpec(formats=("png",), dpi={"png": 72})
        )
        assert [p.suffix for p in written] == [".png"]

    def test_metadata_is_accepted_by_both_backends(self, tmp_path):
        """The PDF and PNG backends take different key sets and raise on unknown ones."""
        fig, _ = plt.subplots()
        written = save_figure(
            fig, "p", tmp_path, metadata=figure_metadata(model="m", trait="t")
        )
        assert len(written) == 2

    def test_metadata_carries_a_timestamp(self):
        assert "created" in figure_metadata(model="m")

    def test_close_closes(self, tmp_path):
        fig, _ = plt.subplots()
        number = fig.number
        save_figure(fig, "p", tmp_path, spec=SaveSpec(formats=("png",), close=True))
        assert not plt.fignum_exists(number)
