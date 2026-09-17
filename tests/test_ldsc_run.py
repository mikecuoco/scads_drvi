"""enrich.run: one LdscRun instead of repeating bfile/w_ld_chr/overlap_annot/binary on
every run_ldsc call."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scads_drvi.enrich.binary import ensure_ldsc
from scads_drvi.enrich.run import LdscRun


def fake_binary(path: Path, version: str = "0.5.0") -> Path:
    """A stand-in that answers --version, so resolution can be tested offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "ldsc ' + version + '"; exit 0; fi\n'
        'echo "$@"\n'
    )
    path.chmod(0o755)
    return path


def write_results(path, z, *, extra_rows=0):
    """One .results file. Row 0 is the factor's own annotation, as LDSC writes it."""
    z_col = "Coefficient_z-score"
    rows = [{"Category": f"{path.stem}L2_0", z_col: z, "Coefficient": 1.0}]
    for i in range(extra_rows):
        rows.append({"Category": f"baseline_{i}", z_col: 0.0, "Coefficient": 0.0})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture
def run(tmp_path):
    return LdscRun(
        binary="ldsc",
        bfile="1000G.EUR.QC.1",
        ld_wind_cm=1,
        w_ld_chr="weights.",
        overlap_annot=True,
        dry_run=True,
        log_fn=None,
    )


class TestL2:
    def test_uses_instance_defaults(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.1.annot.gz", "ld/k1.1")
        (argv,) = seen
        assert argv == (
            "$ ldsc l2 --python-compat --bfile 1000G.EUR.QC.1 "
            "--annot annot/k1.1.annot.gz --ld-wind-cm 1 --out ld/k1.1"
        )

    def test_bfile_override_wins(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.2.annot.gz", "ld/k1.2", bfile="1000G.EUR.QC.2")
        assert "--bfile 1000G.EUR.QC.2" in seen[0]
        assert "1000G.EUR.QC.1" not in seen[0]

    def test_no_bfile_anywhere_is_refused(self):
        run = LdscRun(binary="ldsc", dry_run=True)
        with pytest.raises(ValueError, match="no bfile"):
            run.l2("annot/k1.1.annot.gz", "ld/k1.1")

    def test_extra_options_pass_through(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.1.annot.gz", "ld/k1.1", print_snps="w_hm3.snplist")
        assert "--print-snps w_hm3.snplist" in seen[0]


class TestH2:
    def test_uses_instance_defaults(self, run):
        seen = []
        run.log_fn = seen.append
        run.h2("trait.sumstats.gz", "ld/k1.", "results/trait/k1")
        (argv,) = seen
        # h2 never gets --python-compat: it is an l2-only flag.
        assert "--python-compat" not in argv
        assert argv == (
            "$ ldsc h2 --h2 trait.sumstats.gz --ref-ld-chr ld/k1. "
            "--w-ld-chr weights. --overlap-annot --out results/trait/k1"
        )

    def test_w_ld_chr_and_overlap_annot_can_be_overridden(self, run):
        seen = []
        run.log_fn = seen.append
        run.h2(
            "trait.sumstats.gz",
            "ld/k1.",
            "results/trait/k1",
            w_ld_chr="other_weights.",
            overlap_annot=False,
        )
        argv = seen[0]
        assert "--w-ld-chr other_weights." in argv
        assert "--overlap-annot" not in argv

    def test_no_w_ld_chr_anywhere_is_refused(self):
        run = LdscRun(binary="ldsc", dry_run=True)
        with pytest.raises(ValueError, match="no w_ld_chr"):
            run.h2("trait.sumstats.gz", "ld/k1.", "results/trait/k1")


class TestEnsure:
    def test_wraps_ensure_ldsc(self, tmp_path):
        binary = fake_binary(tmp_path / "ldsc")
        run = LdscRun.ensure(explicit=binary, bfile="1000G.EUR.QC.1", dry_run=True)
        assert run.binary == ensure_ldsc(explicit=binary)
        assert run.bfile == "1000G.EUR.QC.1"


class TestReadResults:
    def test_delegates_to_ldsc_read_results(self, tmp_path):
        annot2dim = {"k1": "dim_0", "k2": "dim_1"}
        root = tmp_path / "results"
        (root / "t1").mkdir(parents=True)
        write_results(root / "t1" / "k1.results", 4.0)
        write_results(root / "t1" / "k2.results", 1.0)

        run = LdscRun(binary="ldsc")
        frame = run.read_results(root, traits=["t1"], annot2dim=annot2dim)
        assert len(frame) == 2
        assert set(frame["dim"]) == {"dim_0", "dim_1"}
