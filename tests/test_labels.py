"""labels: three names per factor, and only one of them may index a matrix."""

from __future__ import annotations

import pandas as pd
import pytest

from scads_drvi.labels import load_labels, read_factor_map


def write_factor_map(path, rows):
    """rows: (dim, kept, annot_index, vanished)"""
    frame = pd.DataFrame(
        [
            {
                "dim": dim,
                "vanished": vanished,
                "kept": kept,
                "drop_reason": "" if kept else "annot_too_small",
                "annot_index": annot,
            }
            for dim, kept, annot, vanished in rows
        ]
    )
    frame.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def simple_map(tmp_path):
    return write_factor_map(
        tmp_path / "factor_map.tsv",
        [
            ("dim_0", False, None, False),
            ("dim_1", True, 1, False),
            ("dim_2", True, 2, False),
            ("dim_3", False, None, True),
            ("dim_4", True, 3, False),
        ],
    )


class TestReadFactorMap:
    def test_string_booleans_round_trip(self, simple_map):
        frame = read_factor_map(simple_map)
        assert frame["kept"].dtype == bool
        assert frame["kept"].tolist() == [False, True, True, False, True]

    def test_annot_index_is_nullable_integer(self, simple_map):
        frame = read_factor_map(simple_map)
        assert str(frame["annot_index"].dtype) == "Int64"
        assert frame["annot_index"].isna().sum() == 2

    def test_missing_file_says_what_writes_it(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="annotation-building stage"):
            read_factor_map(tmp_path / "absent.tsv")

    def test_missing_columns_are_named(self, tmp_path):
        path = tmp_path / "bad.tsv"
        pd.DataFrame({"dim": ["dim_0"]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="annot_index"):
            read_factor_map(path)


class TestLoadLabels:
    def test_kept_factors_only_in_annot_order(self, simple_map):
        labels = load_labels(simple_map, model="m")
        assert labels.kept_dims == ("dim_1", "dim_2", "dim_4")
        assert labels.annot2dim == {"k1": "dim_1", "k2": "dim_2", "k3": "dim_4"}
        assert labels.n_kept == 3 and len(labels) == 3

    def test_no_half_map_means_not_split(self, simple_map):
        labels = load_labels(simple_map, model="m")
        assert labels.is_split is False
        assert labels.display_label("dim_1") == "dim_1"

    def test_all_dropped_is_an_error(self, tmp_path):
        path = write_factor_map(
            tmp_path / "fm.tsv", [("dim_0", False, None, False)]
        )
        with pytest.raises(ValueError, match="keeps no factors"):
            load_labels(path, model="m")

    def test_duplicate_annot_index_is_refused(self, tmp_path):
        path = write_factor_map(
            tmp_path / "fm.tsv",
            [("dim_0", True, 1, False), ("dim_1", True, 1, False)],
        )
        with pytest.raises(ValueError, match="bijection"):
            load_labels(path, model="m")

    @pytest.mark.parametrize(
        "style,expected",
        [("dim", "dim_1"), ("dr", "DR_1"), ("lsi", "LSI_2")],
    )
    def test_display_styles(self, simple_map, style, expected):
        labels = load_labels(simple_map, model="m", style=style)
        assert labels.display_label("dim_1") == expected

    def test_lsi_style_is_one_based(self, simple_map):
        """dim_1 is the second component; calling it LSI_1 has confused readers."""
        labels = load_labels(simple_map, model="m", style="lsi")
        assert labels.display_label("dim_1") == "LSI_2"


class TestSplitContract:
    @pytest.fixture
    def split(self, tmp_path):
        fm = write_factor_map(
            tmp_path / "factor_map.tsv",
            [("dim_0", True, 1, False), ("dim_1", True, 2, False)],
        )
        hm = tmp_path / "half_map.tsv"
        pd.DataFrame(
            {
                "annot_dim": ["dim_0", "dim_1"],
                "source_dim": ["dim_47", "dim_47"],
                "half": ["pos", "neg"],
            }
        ).to_csv(hm, sep="\t", index=False)
        return load_labels(fm, model="m", half_map=hm)

    def test_display_names_the_source_and_direction(self, split):
        assert split.is_split is True
        assert split.display_label("dim_0") == "dim_47/pos"
        assert split.display_label("dim_1") == "dim_47/neg"

    def test_two_columns_can_share_a_source_dimension(self, split):
        assert split.half["dim_0"][0] == split.half["dim_1"][0] == "dim_47"

    def test_display_label_round_trips_through_index_dim(self, split):
        for dim in split.kept_dims:
            assert split.index_dim(split.display_label(dim)) == dim

    def test_malformed_half_map_names_the_missing_columns(self, tmp_path):
        fm = write_factor_map(tmp_path / "fm.tsv", [("dim_0", True, 1, False)])
        hm = tmp_path / "hm.tsv"
        pd.DataFrame({"annot_dim": ["dim_0"]}).to_csv(hm, sep="\t", index=False)
        with pytest.raises(ValueError, match="half"):
            load_labels(fm, model="m", half_map=hm)


class TestIndexResolution:
    def test_index_dim_accepts_all_three_names(self, simple_map):
        labels = load_labels(simple_map, model="m")
        assert labels.index_dim("k1") == "dim_1"
        assert labels.index_dim("dim_1") == "dim_1"

    def test_unknown_name_is_explained(self, simple_map):
        labels = load_labels(simple_map, model="m")
        with pytest.raises(KeyError, match="not a factor"):
            labels.index_dim("nonsense")

    def test_assert_index_dims_accepts_real_columns(self, simple_map):
        labels = load_labels(simple_map, model="m")
        labels.assert_index_dims(["dim_1", "dim_2"])

    def test_assert_index_dims_rejects_an_annotation_name(self, simple_map):
        labels = load_labels(simple_map, model="m")
        with pytest.raises(KeyError, match="annotation name, not a column name"):
            labels.assert_index_dims(["k1"])

    def test_assert_index_dims_rejects_a_display_label(self, tmp_path):
        """The hazard this whole module exists for: under a split contract a display
        label and a column name are both 'dim_'-shaped strings."""
        fm = write_factor_map(tmp_path / "fm.tsv", [("dim_0", True, 1, False)])
        hm = tmp_path / "hm.tsv"
        pd.DataFrame(
            {"annot_dim": ["dim_0"], "source_dim": ["dim_47"], "half": ["neg"]}
        ).to_csv(hm, sep="\t", index=False)
        labels = load_labels(fm, model="m", half_map=hm)
        with pytest.raises(KeyError, match="display label, not a column name"):
            labels.assert_index_dims(["dim_47/neg"])

    def test_display_labels_is_order_preserving(self, simple_map):
        labels = load_labels(simple_map, model="m")
        dims = ["dim_4", "dim_1"]
        assert labels.display_labels(dims) == dims
