"""Widening a thin annotation into the form --overlap-annot will read."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from scads_drvi.enrich.annotations import (  # noqa: E402
    IDENTIFIER_COLUMNS,
    force_decimal,
    is_full_annot,
    read_bim,
    to_full_annot,
    write_full_annot,
)


def make_bim(n: int) -> pd.DataFrame:
    """A bim whose CM column is integral for its first rows and fractional after --
    the shape that trips a reader inferring dtype from a leading sample.
    """
    cm = np.zeros(n)
    cm[n // 2 :] = np.linspace(0.0001, 1.5, n - n // 2)
    return pd.DataFrame(
        {
            "CHR": np.full(n, 22),
            "SNP": [f"rs{1000 + i}" for i in range(n)],
            "CM": cm,
            "BP": np.arange(10_000, 10_000 + n),
            "A1": ["A"] * n,
            "A2": ["G"] * n,
        }
    )


def make_thin(n: int, names=("k1", "k2")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({name: rng.integers(0, 2, n).astype(float) for name in names})


class TestReadBim:
    def test_preserves_file_order(self, tmp_path):
        bim = make_bim(20)
        # Shuffle so a reader that sorted would be caught.
        shuffled = bim.sample(frac=1.0, random_state=1).reset_index(drop=True)
        path = tmp_path / "x.bim"
        shuffled.to_csv(path, sep="\t", header=False, index=False)

        got = read_bim(path)
        assert list(got["SNP"]) == list(shuffled["SNP"])

    def test_empty_bim_is_an_error(self, tmp_path):
        # pandas raises EmptyDataError before read_bim's own check can fire; either way
        # the caller must not receive an empty frame and carry on.
        path = tmp_path / "empty.bim"
        path.write_text("")
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            read_bim(path)


class TestToFullAnnot:
    def test_prefixes_the_identifier_columns_in_reference_order(self):
        bim, thin = make_bim(10), make_thin(10)
        full = to_full_annot(thin, bim)
        assert tuple(full.columns[:4]) == IDENTIFIER_COLUMNS
        assert list(full.columns[4:]) == ["k1", "k2"]
        assert is_full_annot(full.columns)

    def test_combines_positionally_not_by_key(self):
        bim, thin = make_bim(10), make_thin(10)
        full = to_full_annot(thin, bim)
        # Row i of the annotation must still be row i of the bim.
        assert list(full["SNP"]) == list(bim["SNP"])
        assert list(full["BP"]) == list(bim["BP"])
        np.testing.assert_array_equal(full["k1"].to_numpy(), thin["k1"].to_numpy())

    def test_a_thin_annotation_shorter_than_the_bim_is_refused(self):
        # This is the HapMap3-subset mistake: it would put the frequency mask on the
        # wrong variants, and nothing downstream would report it.
        bim, thin = make_bim(10), make_thin(7)
        with pytest.raises(ValueError, match="in bim order"):
            to_full_annot(thin, bim)

    def test_widening_twice_is_refused(self):
        bim, thin = make_bim(10), make_thin(10)
        once = to_full_annot(thin, bim)
        with pytest.raises(ValueError, match="already carries"):
            to_full_annot(once, bim)

    def test_index_of_the_thin_frame_does_not_leak(self):
        bim, thin = make_bim(10), make_thin(10)
        thin.index = [f"v{i}" for i in range(10)]
        full = to_full_annot(thin, bim)
        assert full["k1"].notna().all()
        assert len(full) == 10


class TestForceDecimal:
    def test_integer_columns_become_float(self):
        frame = pd.DataFrame({"CHR": [22, 22], "SNP": ["a", "b"], "CM": [0, 0], "k1": [1, 0]})
        out = force_decimal(frame)
        assert out["CM"].dtype == np.float64
        assert out["k1"].dtype == np.float64

    def test_identifiers_that_are_genuinely_integral_are_left_alone(self):
        frame = pd.DataFrame({"CHR": [22, 22], "SNP": ["a", "b"], "CM": [0, 0], "k1": [1, 0]})
        out = force_decimal(frame)
        assert out["CHR"].dtype == frame["CHR"].dtype
        assert list(out["SNP"]) == ["a", "b"]

    def test_values_are_unchanged(self):
        frame = pd.DataFrame({"CM": [0, 1], "k1": [1, 0]})
        out = force_decimal(frame, ["CM", "k1"])
        np.testing.assert_allclose(out["CM"].to_numpy(), [0.0, 1.0])
        np.testing.assert_allclose(out["k1"].to_numpy(), [1.0, 0.0])

    def test_an_unknown_column_is_an_error(self):
        with pytest.raises(KeyError):
            force_decimal(pd.DataFrame({"a": [1]}), ["nope"])


class TestWriteFullAnnot:
    def test_writes_a_gzipped_table_a_reader_can_type_unambiguously(self, tmp_path):
        bim, thin = make_bim(400), make_thin(400)
        dest = write_full_annot(tmp_path / "annot.22.annot.gz", thin, bim)

        with gzip.open(dest, "rt") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            first = fh.readline().rstrip("\n").split("\t")
        assert tuple(header[:4]) == IDENTIFIER_COLUMNS

        # The point of decimal=True: CM is 0 for the leading rows, and must still be
        # written so that no sample of rows can be read as integral.
        cm = first[header.index("CM")]
        assert "." in cm, f"CM written as {cm!r}, which a reader may infer as integer"

    def test_round_trips(self, tmp_path):
        bim, thin = make_bim(50), make_thin(50)
        dest = write_full_annot(tmp_path / "a.annot.gz", thin, bim)
        back = pd.read_csv(dest, sep="\t")
        assert list(back.columns) == list(IDENTIFIER_COLUMNS) + ["k1", "k2"]
        np.testing.assert_allclose(back["k1"].to_numpy(), thin["k1"].to_numpy())
        assert list(back["SNP"]) == list(bim["SNP"])

    def test_uncompressed_when_the_name_says_so(self, tmp_path):
        bim, thin = make_bim(5), make_thin(5)
        dest = write_full_annot(tmp_path / "a.annot", thin, bim)
        assert dest.read_text().startswith("CHR\tBP\tSNP\tCM")

    def test_creates_missing_parents(self, tmp_path):
        bim, thin = make_bim(5), make_thin(5)
        dest = write_full_annot(tmp_path / "deep" / "er" / "a.annot.gz", thin, bim)
        assert dest.exists()
