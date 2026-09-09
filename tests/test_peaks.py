"""io.peaks: three accepted spellings in, one canonical form out."""

from __future__ import annotations

import pytest

from scads_drvi.io.peaks import PEAK_RE, format_peak, normalize_peak_name, parse_peak


@pytest.mark.parametrize(
    "text",
    ["chr1:100-200", "chr1_100_200", "chr1-100-200"],
)
def test_every_accepted_spelling_parses_the_same(text):
    assert parse_peak(text) == ("chr1", 100, 200)


@pytest.mark.parametrize(
    "text",
    ["chr1:100-200", "chr1_100_200", "chr1-100-200"],
)
def test_every_accepted_spelling_normalises_to_one_form(text):
    assert normalize_peak_name(text) == "chr1:100-200"


def test_chr_prefix_is_added_when_absent():
    assert normalize_peak_name("1:100-200") == "chr1:100-200"
    assert normalize_peak_name("1_100_200") == "chr1:100-200"


def test_canonical_form_matches_the_validation_pattern():
    for text in ("chr1:100-200", "1_100_200", "chrX:5-6", "chr1-100-200"):
        assert PEAK_RE.match(normalize_peak_name(text))


def test_scaffold_names_with_dots_are_accepted():
    assert normalize_peak_name("chrUn_GL000220.1:5-9") == "chrUn_GL000220.1:5-9"
    assert PEAK_RE.match("chrUn_GL000220.1:5-9")


@pytest.mark.parametrize(
    "text",
    ["", "not a peak", "chr1", "chr1:abc-200", "chr1:100", "gene_symbol", "chr1:-5-9"],
)
def test_non_peaks_return_none_rather_than_raising(text):
    """Callers partition a mixed index; aborting on the first non-peak is useless."""
    assert parse_peak(text) is None
    assert normalize_peak_name(text) is None


def test_whitespace_is_tolerated():
    assert normalize_peak_name("  chr1:100-200\n") == "chr1:100-200"


def test_non_string_input_is_coerced():
    class Weird:
        def __str__(self):
            return "chr2:1-2"

    assert normalize_peak_name(Weird()) == "chr2:1-2"
    assert normalize_peak_name(None) is None


def test_dash_form_requires_a_chr_prefix():
    """'a-1-2' is ambiguous: a sequence name may itself contain a dash, so the dash
    spelling is anchored on 'chr' and anything else is left unparsed."""
    assert parse_peak("a-1-2") is None
    assert parse_peak("chra-1-2") == ("chra", 1, 2)


def test_format_peak_round_trips():
    assert format_peak("chr3", 7, 9) == "chr3:7-9"
    assert format_peak("3", 7, 9) == "chr3:7-9"
    assert parse_peak(format_peak("3", 7, 9)) == ("chr3", 7, 9)


def test_zero_and_large_coordinates():
    assert parse_peak("chr1:0-1") == ("chr1", 0, 1)
    assert parse_peak("chr1:0-248956422") == ("chr1", 0, 248956422)
