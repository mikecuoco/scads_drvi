"""Where an analysis lives.

The package derives no path on its own. A caller builds one :class:`Project`, and every
public function takes it. Each field defaults relative to ``root`` and can be overridden
individually, so the defaults are a convenience rather than an assumption -- there is no
capsule, conda-prefix, scheduler or dataset knowledge anywhere in this module.

The directory layout below is the *method's* contract layout (loadings, factors, factor
map, LD-score results). It is a default, not a requirement: pass explicit paths to
:class:`Project` if an analysis is laid out differently.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

__all__ = ["Project", "ROOT_ENV_VAR"]

#: Environment variable consulted by :meth:`Project.from_env`.
ROOT_ENV_VAR = "SCADS_DRVI_ROOT"

# Path-valued fields that default to something under `root` when left as None.
_DERIVED = ("data", "fits", "enrich", "annotations", "figures")


@dataclass(frozen=True)
class Project:
    """Paths and run-wide choices for one analysis.

    Only ``root`` is required. Every other path is filled in relative to it unless
    given explicitly::

        proj = Project(root="/path/to/analysis")
        proj = Project(root="/path/to/analysis", enrich="/scratch/enrich")

    ``fit`` and ``traits`` carry no defaults on purpose: a fit directory name and a set
    of GWAS traits are properties of a particular analysis, and guessing either would
    silently point the pipeline at the wrong inputs.
    """

    root: Path
    data: Path | None = None
    fits: Path | None = None
    enrich: Path | None = None
    annotations: Path | None = None
    figures: Path | None = None
    ldsc_ref: Path | None = None
    ldsc_bin: Path | None = None
    fit: str | None = None
    traits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # frozen dataclass: fill derived fields through object.__setattr__.
        set_ = object.__setattr__
        set_(self, "root", Path(self.root).expanduser().resolve())

        if self.data is None:
            set_(self, "data", self.root / "data")
        if self.fits is None:
            set_(self, "fits", self.data / "factorize")
        if self.enrich is None:
            set_(self, "enrich", self.data / "enrich")
        if self.annotations is None:
            set_(self, "annotations", self.data / "annotations")
        if self.figures is None:
            set_(self, "figures", self.root / "results" / "figures")

        for name in (*_DERIVED, "ldsc_ref", "ldsc_bin"):
            value = getattr(self, name)
            if value is not None:
                set_(self, name, Path(value).expanduser())

        set_(self, "traits", tuple(self.traits))

        if self.fit is not None and ("/" in self.fit or self.fit in (".", "..")):
            raise ValueError(
                f"fit must be a single directory name, not a path: {self.fit!r}"
            )

    # -- directories ---------------------------------------------------------

    def fit_dir(self, fit: str | None = None) -> Path:
        """Directory holding one factorization fit."""
        name = fit if fit is not None else self.fit
        if name is None:
            raise ValueError(
                "no fit name: pass fit_dir('<name>') or set Project(fit='<name>')"
            )
        return self.fits / name

    def enrich_dir(self, model: str) -> Path:
        """Directory holding one enrichment arm's outputs."""
        return self.enrich / model

    def annot_dir(self, model: str) -> Path:
        """Directory holding one arm's factorization contract."""
        return self.annotations / model

    def results_dir(self, model: str, trait: str) -> Path:
        """Directory of per-factor LD-score regression results for one trait."""
        return self.enrich_dir(model) / "results" / trait

    def figures_dir(self, model: str, *, mkdir: bool = True) -> Path:
        """Where figures for one arm are written; created on request."""
        out = self.figures / model
        if mkdir:
            out.mkdir(parents=True, exist_ok=True)
        return out

    # -- the factorization contract -----------------------------------------

    def contract(self, model: str, *, fit: str | None = None) -> dict[str, Path]:
        """The contract files for one arm.

        Keys are stable; the paths are defaults, and two levels are deliberately
        distinguished. ``loadings``/``factors``/``latent_stats`` are written once per
        *fit*; ``factor_map``/``half_map``/``annot_stats``/``results`` are written per
        *arm*, because factor selection and annotation sizing are arm decisions.

        A split contract re-derives its own loadings over the split columns, so an arm
        built that way must be given an explicit ``loadings`` path -- the fit-level
        default has one column per latent dimension, not one per annotation column.

        Missing files are not an error here: callers report that with the context of
        what they were trying to read.
        """
        fdir = self.fit_dir(fit)
        edir = self.enrich_dir(model)
        return {
            "loadings": fdir / "topic_loadings.tsv",
            "loadings_npz": fdir / "topic_loadings.npz",
            "factors": fdir / "topic_factors.tsv",
            "latent_stats": fdir / "inspect" / "latent_stats.tsv",
            "fit_meta": fdir / "fit.meta.json",
            "factor_map": edir / "factor_map.tsv",
            "half_map": edir / "half_map.tsv",
            "annot_stats": edir / "annot_stats.tsv",
            "results": edir / "results",
        }

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides) -> "Project":
        """Build from ``$SCADS_DRVI_ROOT``, falling back to the working directory."""
        root = os.environ.get(ROOT_ENV_VAR) or Path.cwd()
        return cls(root=root, **overrides)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Project":
        """Build from a YAML mapping whose keys are this class's field names.

        Unknown keys are ignored rather than rejected: the file this reads is often a
        pipeline config that carries much more than paths.
        """
        import yaml

        path = Path(path).expanduser().resolve()
        with path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path} does not contain a YAML mapping")

        # A nested `paths:` block is a common shape; merge it over the top level.
        merged = {**raw, **(raw.get("paths") or {})}
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in merged.items() if k in known}
        kwargs.setdefault("root", path.parent)
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- misc ----------------------------------------------------------------

    def replace(self, **changes) -> "Project":
        """A copy with some fields changed."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        """Plain dict with paths stringified, for a run record."""
        out = asdict(self)
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in out.items()
        }
