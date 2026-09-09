"""Project derives every path from `root`, and the package hardcodes none."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import iter_source_files, iter_string_constants

from scads_drvi.config import ROOT_ENV_VAR, Project

# Absolute paths that would tie the package to one machine. A leading "/" alone is too
# blunt -- "/proc/mounts" is legitimate -- so match rooted paths with a second segment.
_ABSOLUTE = re.compile(r"^/(?!proc/|sys/|dev/|usr/|bin/|tmp/?$|$)[A-Za-z0-9_.-]+/")


def test_no_absolute_path_literals():
    """No module may carry an absolute path to somebody's filesystem."""
    offenders = []
    for path in iter_source_files():
        for lineno, value in iter_string_constants(path):
            if _ABSOLUTE.match(value):
                offenders.append(f"{path.name}:{lineno}: {value!r}")
    assert not offenders, "absolute path literals found:\n" + "\n".join(offenders)


def test_defaults_derive_from_root(tmp_path):
    proj = Project(root=tmp_path)
    assert proj.root == tmp_path.resolve()
    assert proj.data == proj.root / "data"
    assert proj.fits == proj.data / "factorize"
    assert proj.enrich == proj.data / "enrich"
    assert proj.annotations == proj.data / "annotations"
    assert proj.figures == proj.root / "results" / "figures"


def test_each_field_is_independently_overridable(tmp_path):
    elsewhere = tmp_path / "scratch" / "enrich"
    proj = Project(root=tmp_path, enrich=elsewhere)
    assert proj.enrich == elsewhere
    # overriding one derived path must not disturb its siblings
    assert proj.annotations == proj.data / "annotations"


def test_root_is_resolved_and_expanded(tmp_path):
    nested = tmp_path / "a" / ".." / "b"
    nested.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    assert Project(root=nested).root == (tmp_path / "b").resolve()


def test_fit_dir_requires_a_name(tmp_path):
    proj = Project(root=tmp_path)
    with pytest.raises(ValueError, match="no fit name"):
        proj.fit_dir()
    assert proj.fit_dir("some_fit") == proj.fits / "some_fit"
    assert Project(root=tmp_path, fit="some_fit").fit_dir() == proj.fits / "some_fit"


def test_fit_must_be_a_bare_name(tmp_path):
    with pytest.raises(ValueError, match="single directory name"):
        Project(root=tmp_path, fit="nested/fit")


def test_contract_keys_are_stable(tmp_path):
    proj = Project(root=tmp_path, fit="a_fit")
    contract = proj.contract("arm")
    assert set(contract) == {
        "loadings", "loadings_npz", "factors", "latent_stats", "fit_meta",
        "factor_map", "half_map", "annot_stats", "results",
    }
    assert all(isinstance(v, Path) for v in contract.values())


def test_contract_separates_fit_level_from_arm_level(tmp_path):
    """Loadings are written once per fit; factor selection is per arm. Conflating them
    is how an arm ends up reading another arm's factor map."""
    proj = Project(root=tmp_path, fit="a_fit")
    contract = proj.contract("arm")
    assert contract["loadings"].is_relative_to(proj.fit_dir())
    assert contract["latent_stats"].is_relative_to(proj.fit_dir())
    assert contract["factor_map"].is_relative_to(proj.enrich_dir("arm"))
    assert contract["results"].is_relative_to(proj.enrich_dir("arm"))


def test_contract_fit_can_be_overridden_per_call(tmp_path):
    proj = Project(root=tmp_path, fit="a_fit")
    other = proj.contract("arm", fit="b_fit")
    assert other["loadings"].is_relative_to(proj.fits / "b_fit")
    # the arm-level half is unaffected by which fit was named
    assert other["factor_map"] == proj.contract("arm")["factor_map"]


def test_figures_dir_creates_on_request(tmp_path):
    proj = Project(root=tmp_path)
    assert not (proj.figures / "arm").exists()
    made = proj.figures_dir("arm")
    assert made.is_dir()
    assert proj.figures_dir("other", mkdir=False).exists() is False


def test_traits_normalise_to_a_tuple(tmp_path):
    assert Project(root=tmp_path, traits=["a", "b"]).traits == ("a", "b")
    assert Project(root=tmp_path).traits == ()


def test_from_env_uses_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    assert Project.from_env().root == tmp_path.resolve()
    monkeypatch.delenv(ROOT_ENV_VAR)
    monkeypatch.chdir(tmp_path)
    assert Project.from_env().root == tmp_path.resolve()


def test_from_yaml_reads_known_keys_and_ignores_the_rest(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fit: a_fit\n"
        "traits: [t1, t2]\n"
        "paths:\n"
        f"  enrich: {tmp_path / 'e'}\n"
        "unrelated_key: 3\n"
    )
    proj = Project.from_yaml(cfg)
    assert proj.fit == "a_fit"
    assert proj.traits == ("t1", "t2")
    assert proj.enrich == tmp_path / "e"
    assert proj.root == tmp_path.resolve()


def test_replace_and_as_dict_round_trip(tmp_path):
    proj = Project(root=tmp_path, fit="f")
    assert proj.replace(fit="g").fit == "g"
    assert proj.fit == "f"  # frozen: the original is untouched
    as_dict = proj.as_dict()
    assert as_dict["root"] == str(tmp_path.resolve())
    assert isinstance(as_dict["root"], str)
