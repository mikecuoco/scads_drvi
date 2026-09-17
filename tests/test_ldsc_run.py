"""enrich.run: LdscRun resolves the binary, builds each subcommand's argv, and runs
it -- one object instead of repeating bfile/w_ld_chr/overlap_annot/binary on every
call."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scads_drvi.enrich.binary import LDSC_ENV_VAR, LDSC_VERSION
from scads_drvi.enrich.run import APPROXIMATE_FLAGS, SUBCOMMANDS, LdscRun


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
        """sketch=200 and snp_level_masking=True are on by default, for speed;
        python_compat defaults off since the binary refuses it together with
        snp-level masking."""
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.1.annot.gz", "ld/k1.1")
        (argv,) = seen
        assert argv == (
            "$ ldsc l2 --bfile 1000G.EUR.QC.1 --annot annot/k1.1.annot.gz "
            "--ld-wind-cm 1 --sketch 200 --snp-level-masking --out ld/k1.1"
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

    def test_sketch_can_be_turned_off_per_call(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.1.annot.gz", "ld/k1.1", sketch=None, snp_level_masking=False)
        argv = seen[0]
        assert "--sketch" not in argv
        assert "--snp-level-masking" not in argv

    def test_sketch_can_be_overridden_per_call(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2("annot/k1.1.annot.gz", "ld/k1.1", sketch=5000)
        assert "--sketch 5000" in seen[0]

    def test_python_compat_and_snp_level_masking_together_is_refused(self, run):
        with pytest.raises(ValueError, match="cannot both be on"):
            run.l2("annot/k1.1.annot.gz", "ld/k1.1", python_compat=True)

    def test_python_compat_works_once_snp_level_masking_is_off(self, run):
        seen = []
        run.log_fn = seen.append
        run.l2(
            "annot/k1.1.annot.gz",
            "ld/k1.1",
            python_compat=True,
            snp_level_masking=False,
            sketch=None,
        )
        assert "--python-compat" in seen[0]

    def test_sketch_requires_allow_approximate_is_derived_not_forgotten(self, run):
        """sketch is one of binary.APPROXIMATE_FLAGS, normally gated behind
        allow_approximate=True on run_ldsc -- l2() must derive that gate from `sketch`
        being set rather than needing a separate switch, or the default above would
        raise instead of running."""
        run.l2("annot/k1.1.annot.gz", "ld/k1.1")  # would raise ValueError if ungated


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
    def test_wraps_resolve_binary(self, tmp_path):
        binary = fake_binary(tmp_path / "ldsc")
        run = LdscRun.ensure(explicit=binary, bfile="1000G.EUR.QC.1", dry_run=True)
        assert run.binary == LdscRun._resolve_binary(explicit=binary)
        assert run.bfile == "1000G.EUR.QC.1"


class TestResolveBinary:
    def test_explicit_wins(self, tmp_path):
        binary = fake_binary(tmp_path / "mine" / "ldsc")
        got = LdscRun._resolve_binary(explicit=binary, cache=tmp_path / "cache")
        assert got.source == "explicit"
        assert Path(got) == binary

    def test_explicit_missing_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no ldsc binary"):
            LdscRun._resolve_binary(explicit=tmp_path / "absent")

    def test_environment_is_next(self, tmp_path, monkeypatch):
        binary = fake_binary(tmp_path / "env" / "ldsc")
        monkeypatch.setenv(LDSC_ENV_VAR, str(binary))
        got = LdscRun._resolve_binary(cache=tmp_path / "cache")
        assert got.source == "environment"

    def test_environment_pointing_nowhere_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setenv(LDSC_ENV_VAR, str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match=LDSC_ENV_VAR):
            LdscRun._resolve_binary(cache=tmp_path / "cache")

    def test_cache_is_used_before_downloading(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LDSC_ENV_VAR, raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        cache = tmp_path / "cache"
        fake_binary(cache / "ldsc")
        got = LdscRun._resolve_binary(cache=cache, allow_download=False)
        assert got.source == "cache"

    def test_no_binary_and_no_download_explains_both_routes(self, tmp_path, monkeypatch):
        monkeypatch.delenv(LDSC_ENV_VAR, raising=False)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(FileNotFoundError, match="cargo install ldsc"):
            LdscRun._resolve_binary(cache=tmp_path / "cache", allow_download=False)


class TestVersionChecking:
    def test_a_mismatched_version_is_refused(self, tmp_path):
        """Silently running a different LD-score implementation than the one recorded
        is how a number becomes unreproducible months later."""
        binary = fake_binary(tmp_path / "old" / "ldsc", version="0.1.2")
        with pytest.raises(OSError, match="pins"):
            LdscRun._resolve_binary(explicit=binary, cache=tmp_path / "cache")

    def test_the_mismatch_can_be_accepted_deliberately(self, tmp_path):
        binary = fake_binary(tmp_path / "old" / "ldsc", version="0.1.2")
        got = LdscRun._resolve_binary(
            explicit=binary, cache=tmp_path / "cache", check_version=False
        )
        assert got.version == "0.1.2"

    def test_an_unparseable_version_is_an_error(self, tmp_path):
        path = tmp_path / "ldsc"
        path.write_text("#!/bin/sh\necho not-a-version\n")
        path.chmod(0o755)
        with pytest.raises(OSError, match="cannot parse a version"):
            LdscRun._resolve_binary(explicit=path, cache=tmp_path / "cache")


class TestBuildCommand:
    def test_python_compat_is_added_to_l2(self):
        argv = LdscRun(binary="ldsc")._build_command("l2", {"bfile": "x"})
        assert "--python-compat" in argv

    def test_python_compat_is_not_added_to_h2(self):
        """It exists only on l2; passing it to h2 would be a hard error from the tool."""
        assert SUBCOMMANDS["h2"] is False
        argv = LdscRun(binary="ldsc")._build_command("h2", {"h2": "x.sumstats.gz"})
        assert "--python-compat" not in argv

    def test_python_compat_can_be_turned_off(self):
        argv = LdscRun(binary="ldsc")._build_command(
            "l2", {"bfile": "x"}, python_compat=False
        )
        assert "--python-compat" not in argv

    def test_underscores_become_dashes(self):
        argv = LdscRun(binary="ldsc")._build_command(
            "h2", {"ref_ld_chr": "p", "w_ld_chr": "q"}
        )
        assert "--ref-ld-chr" in argv and "--w-ld-chr" in argv

    def test_booleans_are_switches_and_none_is_dropped(self):
        argv = LdscRun(binary="ldsc")._build_command(
            "h2", {"overlap_annot": True, "frqfile": None}
        )
        assert "--overlap-annot" in argv
        assert not any(a.startswith("--frqfile") for a in argv)

    def test_false_is_dropped(self):
        argv = LdscRun(binary="ldsc")._build_command("h2", {"overlap_annot": False})
        assert "--overlap-annot" not in argv

    def test_lists_are_comma_joined(self):
        argv = LdscRun(binary="ldsc")._build_command("l2", {"chroms": [1, 2, 22]})
        assert argv[argv.index("--chroms") + 1] == "1,2,22"

    @pytest.mark.parametrize("flag", APPROXIMATE_FLAGS)
    def test_approximate_flags_are_refused_by_default(self, flag):
        """--sketch is a random projection; a fast wrong number must not be reachable
        by accident."""
        with pytest.raises(ValueError, match="change the estimate"):
            LdscRun(binary="ldsc")._build_command("l2", {flag: 200})

    def test_approximate_flags_can_be_opted_into(self):
        argv = LdscRun(binary="ldsc")._build_command(
            "l2", {"sketch": 200}, allow_approximate=True
        )
        assert "--sketch" in argv

    def test_unknown_subcommand_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="expected one of"):
            LdscRun(binary="ldsc")._build_command("nope", {})


class TestRun:
    def test_dry_run_echoes_without_executing(self, tmp_path):
        seen: list[str] = []
        run = LdscRun(binary=tmp_path / "nonexistent", dry_run=True, log_fn=seen.append)
        proc = run._run("l2", {"bfile": "x"}, python_compat=True)
        assert proc.returncode == 0
        assert seen and seen[0].startswith("$ ")
        assert "--python-compat" in seen[0]

    def test_threads_cap_both_pools(self, tmp_path):
        seen: list[str] = []
        run = LdscRun(
            binary=tmp_path / "b", threads=1, dry_run=True, log_fn=seen.append
        )
        run._run("h2", {"h2": "x"}, python_compat=False)
        assert "--rayon-threads 1" in seen[0]
        assert "--polars-threads 1" in seen[0]

    def test_it_actually_runs_a_binary(self, tmp_path):
        binary = fake_binary(tmp_path / "ldsc")
        run = LdscRun(binary=binary, check=True)
        proc = run._run("h2", {"h2": "x"}, python_compat=False)
        assert proc.returncode == 0
        assert "h2" in proc.stdout


@pytest.mark.data
class TestAgainstTheRealBinary:
    """Runs only where the pinned binary is already resolvable."""

    def _resolve(self):
        try:
            return LdscRun.ensure(allow_download=False, dry_run=True)
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"pinned ldsc not available offline: {exc}")

    def test_version_matches_the_pin(self):
        run = self._resolve()
        assert run.binary.version == LDSC_VERSION.lstrip("v")

    def test_l2_advertises_python_compat(self):
        run = self._resolve()
        help_text = subprocess.run(
            [str(run.binary.path), "l2", "--help"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        assert "--python-compat" in help_text

    def test_h2_does_not_advertise_python_compat(self):
        """The asymmetry this module encodes, checked against the tool itself."""
        run = self._resolve()
        help_text = subprocess.run(
            [str(run.binary.path), "h2", "--help"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        assert "--python-compat" not in help_text
        assert "--overlap-annot" in help_text
        assert "--frqfile-chr" in help_text


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
