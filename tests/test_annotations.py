"""Reading a bim's row order."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from scads_drvi.enrich.annotations import read_bim  # noqa: E402


def make_bim(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": np.full(n, 22),
            "SNP": [f"rs{1000 + i}" for i in range(n)],
            "CM": np.linspace(0.0, 1.5, n),
            "BP": np.arange(10_000, 10_000 + n),
            "A1": ["A"] * n,
            "A2": ["G"] * n,
        }
    )


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
