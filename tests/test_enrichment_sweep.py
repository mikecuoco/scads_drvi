"""enrich.sweep: embed + reference panel + sumstats -> the l2/h2 sweep -> results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")

from scads_drvi.enrich.run import LdscRun
from scads_drvi.enrich.sweep import EnrichmentSweep


def write_bim(path, snps):
    """A minimal PLINK .bim: CHR SNP CM BP A1 A2, whitespace-delimited."""
    lines = [f"1\t{snp}\t0.0\t{bp}\tA\tG" for snp, bp in snps]
    Path(path).write_text("\n".join(lines) + "\n")


def write_results(path, z):
    """One .results file. Row 0 is the factor's own annotation, as LDSC writes it."""
    z_col = "Coefficient_z-score"
    row = {"Category": f"{path.stem}L2_0", z_col: z, "Coefficient": 1.0}
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)


def make_embed(vanished=(False, False, True)):
    n, k = 5, len(vanished)
    x = np.zeros((n, k), dtype=np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(
        {"vanished": list(vanished)}, index=[f"dim_{i}" for i in range(k)]
    )
    return ad.AnnData(X=x, obs=obs, var=var)


def stub_annotate(dim, direction, chrom, bim):
    return np.ones(len(bim))


@pytest.fixture
def refs(tmp_path):
    """Two chromosomes' worth of tiny .bim files, at bfile_chr's template."""
    for chrom in (1, 2):
        write_bim(tmp_path / f"ref.{chrom}.bim", [("rs1", 1000), ("rs2", 2000)])
    return str(tmp_path / "ref.{chrom}")


@pytest.fixture
def sweep(tmp_path, refs):
    ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=False, dry_run=True)
    return EnrichmentSweep(
        ldsc_run=ldsc_run,
        embed=make_embed(),
        bfile_chr=refs,
        sumstats={"t1": "t1.sumstats.gz"},
        annotate=stub_annotate,
        chroms=(1, 2),
        directional=True,
        ldscore_dir=tmp_path / "ldscore",
        results_dir=tmp_path / "results",
    )


class TestKeys:
    def test_directional_keys_and_maps(self, sweep):
        assert sweep.keep == ["dim_0", "dim_1"]  # dim_2 dropped: vanished
        assert sweep.annot2dim() == {
            "dim_0_pos": "dim_0", "dim_0_neg": "dim_0",
            "dim_1_pos": "dim_1", "dim_1_neg": "dim_1",
        }
        assert sweep.direction_map() == {
            "dim_0_pos": "pos", "dim_0_neg": "neg",
            "dim_1_pos": "pos", "dim_1_neg": "neg",
        }

    def test_non_directional_keys_and_maps(self, tmp_path, refs):
        ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=False, dry_run=True)
        s = EnrichmentSweep(
            ldsc_run=ldsc_run, embed=make_embed(), bfile_chr=refs,
            sumstats={"t1": "x"}, annotate=stub_annotate, directional=False,
            ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
        )
        assert s.annot2dim() == {"dim_0": "dim_0", "dim_1": "dim_1"}
        assert s.direction_map() == "combined"

    def test_dims_and_directions_can_be_filtered(self, sweep):
        assert sweep.annot2dim(dims=["dim_0"], directions=["pos"]) == {"dim_0_pos": "dim_0"}


class TestRunL2:
    def test_builds_per_chromosome_argv(self, sweep):
        seen = []
        sweep.ldsc_run.log_fn = seen.append
        sweep.run_l2(dims=["dim_0"], directions=["pos"])
        assert len(seen) == 2  # one per chromosome
        for chrom, argv in zip((1, 2), seen, strict=True):
            assert f"--bfile {sweep.bfile_chr.format(chrom=chrom)}" in argv
            assert f"--out {sweep.ldscore_dir}/dim_0_pos.{chrom}" in argv
            assert "--annot" in argv  # a tempfile path -- not asserted verbatim

    def test_skips_when_already_done(self, sweep):
        out = sweep._ld_out("dim_0_pos", 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.l2.ldscore.gz").touch()

        seen = []
        sweep.ldsc_run.log_fn = seen.append
        sweep.run_l2(dims=["dim_0"], directions=["pos"], chroms=[1])
        assert seen == []  # already done, not re-run

    def test_force_reruns(self, sweep):
        out = sweep._ld_out("dim_0_pos", 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.l2.ldscore.gz").touch()

        seen = []
        sweep.ldsc_run.log_fn = seen.append
        sweep.run_l2(dims=["dim_0"], directions=["pos"], chroms=[1], force=True)
        assert len(seen) == 1

    def test_annotation_files_never_persist(self, sweep, tmp_path):
        sweep.run_l2(dims=["dim_0"], directions=["pos"], chroms=[1])
        leftovers = list(Path(tmp_path).rglob("*.annot.gz"))
        assert leftovers == []


class TestRunH2:
    def test_ref_ld_chr_wildcards_and_extra_is_appended(self, sweep):
        sweep.ref_ld_chr_extra = ("baseline.",)
        sweep.frqfile_chr = "freq."
        seen = []
        sweep.ldsc_run.log_fn = seen.append
        sweep.run_h2(dims=["dim_0"], directions=["pos"], traits=["t1"])
        (argv,) = seen
        assert f"--ref-ld-chr {sweep.ldscore_dir}/dim_0_pos.,baseline." in argv
        assert "--frqfile-chr freq." in argv
        assert "--print-coefficients" in argv
        assert f"--out {sweep.results_dir}/t1/dim_0_pos" in argv

    def test_skips_when_already_done(self, sweep):
        out = sweep._h2_out("t1", "dim_0_pos")
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.results").touch()

        seen = []
        sweep.ldsc_run.log_fn = seen.append
        sweep.run_h2(dims=["dim_0"], directions=["pos"], traits=["t1"])
        assert seen == []


class TestReadResults:
    def test_delegates_with_this_sweeps_annot2dim_and_direction(self, sweep):
        for key in ("dim_0_pos", "dim_0_neg", "dim_1_pos", "dim_1_neg"):
            d = sweep.results_dir / "t1"
            d.mkdir(parents=True, exist_ok=True)
            write_results(d / f"{key}.results", 2.0)
        frame = sweep.read_results()
        assert len(frame) == 4
        assert set(frame["dim"]) == {"dim_0", "dim_1"}
        assert set(frame["direction"]) == {"pos", "neg"}


class TestEmbedPath:
    def test_h5ad_path_defaults_dirs_alongside_it(self, tmp_path, refs):
        h5ad_path = tmp_path / "fit" / "embed.h5ad"
        h5ad_path.parent.mkdir(parents=True)
        make_embed().write_h5ad(h5ad_path)

        ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=False, dry_run=True)
        s = EnrichmentSweep(
            ldsc_run=ldsc_run, embed=str(h5ad_path), bfile_chr=refs,
            sumstats={"t1": "x"}, annotate=stub_annotate,
        )
        assert s.ldscore_dir == h5ad_path.parent
        assert s.results_dir == h5ad_path.parent
        assert s.keep == ["dim_0", "dim_1"]

    def test_in_memory_embed_requires_explicit_dirs(self, refs):
        ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=False, dry_run=True)
        with pytest.raises(ValueError, match="must both be given explicitly"):
            EnrichmentSweep(
                ldsc_run=ldsc_run, embed=make_embed(), bfile_chr=refs,
                sumstats={"t1": "x"}, annotate=stub_annotate,
            )


class TestOverlapAnnotRequiresFrqfile:
    def test_raises_up_front(self, tmp_path, refs):
        ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=True, dry_run=True)
        with pytest.raises(ValueError, match="frqfile_chr"):
            EnrichmentSweep(
                ldsc_run=ldsc_run, embed=make_embed(), bfile_chr=refs,
                sumstats={"t1": "x"}, annotate=stub_annotate,
                ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
            )

    def test_passes_with_frqfile_chr_set(self, tmp_path, refs):
        ldsc_run = LdscRun(binary="ldsc", w_ld_chr="weights.", overlap_annot=True, dry_run=True)
        EnrichmentSweep(
            ldsc_run=ldsc_run, embed=make_embed(), bfile_chr=refs,
            sumstats={"t1": "x"}, annotate=stub_annotate, frqfile_chr="freq.",
            ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
        )


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


class TestEnsure:
    def test_defaults_ref_ld_chr_extra_to_the_ukb_reference(
        self, tmp_path, refs, monkeypatch
    ):
        seen_calls = []

        def fake_ensure_baseline_ukb(*, cache=None, allow_download=True):
            seen_calls.append((cache, allow_download))
            return "/cache/baselineLF_v2.2.UKB/baselineLF2.2.UKB."

        monkeypatch.setattr(
            "scads_drvi.enrich.reference.ensure_baseline_ukb", fake_ensure_baseline_ukb
        )
        binary = fake_binary(tmp_path / "ldsc")
        s = EnrichmentSweep.ensure(
            embed=make_embed(), bfile_chr=refs, sumstats={"t1": "x"},
            annotate=stub_annotate, w_ld_chr="weights.", overlap_annot=False,
            explicit=binary, ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
        )
        assert seen_calls == [(None, True)]
        assert s.ref_ld_chr_extra == ("/cache/baselineLF_v2.2.UKB/baselineLF2.2.UKB.",)
        assert s.ldsc_run.sketch == 5000  # UKB scale: LdscRun's own 200 default is too low

    def test_explicit_empty_ref_ld_chr_extra_skips_the_ukb_default(
        self, tmp_path, refs, monkeypatch
    ):
        def fail(*a, **k):
            raise AssertionError("ensure_baseline_ukb should not be called")

        monkeypatch.setattr("scads_drvi.enrich.reference.ensure_baseline_ukb", fail)
        binary = fake_binary(tmp_path / "ldsc")
        s = EnrichmentSweep.ensure(
            embed=make_embed(), bfile_chr=refs, sumstats={"t1": "x"},
            annotate=stub_annotate, w_ld_chr="weights.", overlap_annot=False,
            ref_ld_chr_extra=(), explicit=binary,
            ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
        )
        assert s.ldsc_run.sketch == 200  # 1000G-scale default, unchanged from LdscRun
        assert s.ref_ld_chr_extra == ()

    def test_explicit_sketch_overrides_either_default(self, tmp_path, refs, monkeypatch):
        monkeypatch.setattr(
            "scads_drvi.enrich.reference.ensure_baseline_ukb",
            lambda *, cache=None, allow_download=True: "/cache/stem.",
        )
        binary = fake_binary(tmp_path / "ldsc")
        s = EnrichmentSweep.ensure(
            embed=make_embed(), bfile_chr=refs, sumstats={"t1": "x"},
            annotate=stub_annotate, w_ld_chr="weights.", overlap_annot=False,
            sketch=999, explicit=binary,
            ldscore_dir=tmp_path / "ld", results_dir=tmp_path / "res",
        )
        assert s.ldsc_run.sketch == 999
