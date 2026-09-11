# Getting Started

## Installation

```bash
# standard editable install
pip install -e .

# inside a mamba-solved environment — do NOT let pip re-resolve numpy/torch
pip install -e . --no-deps

# directly from GitHub
pip install 'git+https://github.com/mikecuoco/scads_drvi@main#egg=scads-drvi'
```

:::{note}
If you are working inside a conda/mamba environment with numpy and torch already solved,
always use `--no-deps`. Letting pip re-resolve those packages across a solved prefix will
break the environment.
:::

## Install documentation dependencies

```bash
pip install -e '.[docs]'
cd docs && make html
# open docs/_build/html/index.html
```

## The `Project` object

Every path in the pipeline flows through a single {class}`~scads_drvi.config.Project`
object. Nothing is hard-coded — every directory is either supplied by the caller or
derived from a `root`:

```python
import scads_drvi as sd

proj = sd.Project(
    root="/path/to/analysis",
    fit="my_fit",
    traits=("trait_a", "trait_b"),
)
```

Default directory layout under `root`:

| `Project` field | default |
|---|---|
| `data` | `root/data` |
| `fits` | `data/factorize` |
| `enrich` | `data/enrich` |
| `annotations` | `data/annotations` |
| `figures` | `root/results/figures` |

`fit` and `traits` have **no defaults on purpose** — guessing either silently points
the pipeline at the wrong inputs.

### Load from a YAML file

```python
proj = sd.Project.from_yaml("/path/to/project.yaml")
```

```yaml
# project.yaml
root: /path/to/analysis
fit: my_fit
traits:
  - trait_a
  - trait_b
```

### Load from environment variables

```bash
export SCADS_DRVI_ROOT=/path/to/analysis
```

```python
proj = sd.Project.from_env(fit="my_fit", traits=("trait_a", "trait_b"))
```

## Reading a finished run

Once a run has completed you can read everything through
{func}`~scads_drvi.io.artifacts.load_interpretation`:

```python
from scads_drvi.io.artifacts import load_interpretation

interp = load_interpretation(
    proj,
    model="my_arm",
    obs_path=proj.data / "matrix.h5ad",
    obs_columns=["cell_type", "tissue", "donor"],
    derived={"peak_fraction": ("reads_in_peaks", "total_reads")},
    umap_path=proj.enrich_dir("my_arm") / "umap.tsv",
)

interp.traits            # ("trait_a", "trait_b")
interp.n_cells           # 1_263_026
interp.labels.n_kept     # 124

results = interp.for_trait("trait_a")   # per-factor results DataFrame
```

## Computing cell scores

```python
from scads_drvi.io.artifacts import read_loadings
from scads_drvi.scores.cell import cs_from_z

loadings = read_loadings(
    proj.contract("my_arm")["loadings"],
    npz=proj.contract("my_arm")["loadings_npz"],
    dims=list(interp.labels.kept_dims),
)
scores = cs_from_z(
    loadings, results, model="my_arm", trait="trait_a", labels=interp.labels
)

scores.null    # 0.0  — read this; never hardcode it beside an axis
scores.label   # "$CS_i$ (z-weighted loading sum)"
```

## Import surface

`import scads_drvi` loads nothing heavier than the standard library. Heavy dependencies
are confined by directory and imported inside functions:

| module | needs |
|---|---|
| `factorize/model.py`, `factorize/multigpu.py` | `torch`, `scvi-tools` (function-local) |
| `viz/` | `matplotlib`, `seaborn` (function-local) |
| `io/h5ad.py` | `h5py` at module scope |
| `io/artifacts.py` | `h5py`, function-local |
| everything else | `numpy` / `pandas` / `scipy` / `pyyaml` |

This matters because the environment that runs enrichment stages has no torch, matplotlib,
seaborn or h5py, and must still import and use the loaders, statistics, and scoring.

## Running the tests

```bash
pytest
```

CI runs on three dependency stacks:

1. numpy 2 / pandas 3, no optional dependencies
2. Declared floors (numpy 1.26 / pandas 2.1)
3. Current stack with h5py and plotting

If you vendor this package next to another test suite, collect the two separately.
