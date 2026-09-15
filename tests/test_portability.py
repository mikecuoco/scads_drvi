"""The whole pipeline surface on a dataset that shares no vocabulary with SEA-AD.

The genericity scan (``test_generic.py``) proves the package never *names* a
dataset-specific concept. That is necessary but not sufficient: a package can be free of
forbidden strings and still assume a column exists, an index is shaped a certain way, or
seven regions are called something.

So this builds a complete synthetic analysis from scratch -- a plant single-cell
experiment, with tissues, cultivars and agronomic traits -- and runs it end to end:
h5ad obs -> the cells x K embed -> write_result -> factor selection -> LDSC results ->
attach_enrich_results -> BH -> per-cell scores -> aggregation -> figures. Nothing here
is renamed from the real analysis; the point is that no name matches.

If the package ever grows an assumption about the real dataset, this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scads_drvi.enrich.config import (  # noqa: E402
    kept_dims,
    latent_stats_from_embed,
    select_factors,
)
from scads_drvi.enrich.embed import (  # noqa: E402
    attach_enrich_results,
    directional_loadings,
    write_result,
)
from scads_drvi.enrich.ldsc import read_results  # noqa: E402
from scads_drvi.pl.enrichment import (  # noqa: E402
    covariate_audit,
    grouped_landscape,
    heritability_landscape,
    score_by_group,
    trait_concordance,
)
from scads_drvi.pl.frugal import box_stats  # noqa: E402
from scads_drvi.pl.save import save_figure  # noqa: E402
from scads_drvi.pl.umap import umap_categorical, umap_continuous  # noqa: E402
from scads_drvi.scores.aggregate import (  # noqa: E402
    block_order,
    eta_squared,
    group_matrix,
    summarize_by,
)
from scads_drvi.scores.cell import ScoreKind, cs_from_z  # noqa: E402
from scads_drvi.stats import significant  # noqa: E402

# A vocabulary with no overlap with the real analysis.
TISSUES = ("leaf", "root", "stem", "flower")
LINEAGES = ("mesophyll", "epidermis", "vasculature")
FINE_TYPES = {
    "mesophyll": ("meso_1", "meso_2", "meso_3"),
    "epidermis": ("epi_1", "epi_2"),
    "vasculature": ("vasc_1", "vasc_2", "vasc_3"),
}
CULTIVARS = ("cv_alpha", "cv_beta", "cv_gamma")
TRAITS = ("plant_height", "grain_yield")
N_CELLS = 3000
N_FACTORS = 12
N_KEPT = 9
ARM = "assembly_v3_topfrac"


class _FakeModel:
    """Stands in for a trained ``scvi.external.DRVI``: only the two methods the
    cells x K embed construction needs -- ``get_latent_representation`` and DRVI's own
    ``set_latent_dimension_stats``, driven by a fixed signed latent matrix and a fixed
    kept/vanished split, so the rest of this test can exercise the real pipeline
    functions against it."""

    def __init__(self, z, vanished):
        self._z = z
        self._vanished = np.asarray(vanished, dtype=bool)

    def get_latent_representation(self, adata=None, indices=None, batch_size=128):
        out = self._z
        if indices is not None:
            out = out[np.asarray(indices)]
        return out

    def set_latent_dimension_stats(
        self, embed, adata=None, datamodule=None, vanished_threshold=0.5
    ):
        embed.var["vanished"] = self._vanished
        embed.var["order"] = np.arange(embed.n_vars)
        embed.var["title"] = [f"DR {i + 1}" for i in range(embed.n_vars)]


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    """A complete synthetic run on disk, in the shape the real pipeline produces."""
    root = tmp_path_factory.mktemp("plant_atac")
    rng = np.random.default_rng(0)

    fine = [t for lineage in LINEAGES for t in FINE_TYPES[lineage]]
    lineage_of = {t: lin for lin in LINEAGES for t in FINE_TYPES[lin]}

    cell_ids = [f"plate{i % 5}:bc{i:05d}" for i in range(N_CELLS)]
    fine_vals = rng.choice(fine, size=N_CELLS)
    tissue_vals = rng.choice(TISSUES, size=N_CELLS)
    cultivar_vals = rng.choice(CULTIVARS, size=N_CELLS)
    reads = rng.integers(500, 200_000, size=N_CELLS).astype(np.int64)
    in_peaks = (reads * rng.uniform(0.2, 0.8, size=N_CELLS)).astype(np.int64)
    in_peaks[0] = 0
    reads[0] = 0  # a cell with nothing measured, so the derived ratio must be NaN

    # A real customer's obs h5ad, written the way anndata itself writes one -- no
    # hand-rolled h5py encoding, since there is no custom decoder left to stress-test.
    obs_df = pd.DataFrame(
        {
            "fine_type": pd.Categorical(fine_vals),
            "tissue": pd.Categorical(tissue_vals),
            "cultivar": pd.Categorical(cultivar_vals),
            "lineage": pd.Categorical(
                [lineage_of[f] for f in fine_vals], categories=LINEAGES
            ),
            "total_reads": reads,
            "reads_in_peaks": in_peaks,
        },
        index=pd.Index(cell_ids, name="cell_uid"),
    )
    obs_path = root / "matrix.h5ad"
    ad.AnnData(X=np.zeros((N_CELLS, 1), dtype=np.float32), obs=obs_df).write_h5ad(obs_path)

    # -- obs h5ad + a trained model -> the canonical cells x K object -----------------
    raw = ad.read_h5ad(obs_path)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw.obs["peak_fraction"] = np.where(
            raw.obs["total_reads"].to_numpy() > 0,
            raw.obs["reads_in_peaks"].to_numpy() / raw.obs["total_reads"].to_numpy(),
            np.nan,
        )
    z = rng.normal(0.2, 0.6, size=(N_CELLS, N_FACTORS)).astype(np.float32)
    vanished = np.array([False] * N_KEPT + [True] * (N_FACTORS - N_KEPT))
    model = _FakeModel(z, vanished)

    obs_columns = [
        "fine_type", "tissue", "cultivar", "lineage", "total_reads", "peak_fraction",
    ]
    embed = ad.AnnData(
        X=model.get_latent_representation(raw), obs=raw.obs[obs_columns].copy()
    )
    embed.var_names = [f"dim_{i}" for i in range(embed.n_vars)]
    model.set_latent_dimension_stats(embed)
    # A UMAP is set directly from the embed once computed -- no read/reindex step.
    embed.obsm["X_umap"] = rng.normal(size=(N_CELLS, 2)).astype(np.float32)

    path = root / "assembly_v3.h5ad"
    write_result(
        path, embed, provenance={"n_latent": N_FACTORS, "fit_name": "assembly_v3"}
    )

    # -- factor selection, exactly the way the real enrichment stage would ------------
    embed = ad.read_h5ad(path)
    stats = latent_stats_from_embed(embed)
    fmap = select_factors(stats, list(embed.var_names))
    keep = kept_dims(fmap)
    assert len(keep) == N_KEPT
    annot2dim = {f"k{i + 1}": dim for i, dim in enumerate(keep)}

    results_root = root / "results"
    for trait_index, trait in enumerate(TRAITS):
        trait_dir = results_root / trait
        trait_dir.mkdir(parents=True)
        for slot, annot in enumerate(annot2dim):
            z_score = 4.5 - 0.6 * slot + 0.3 * trait_index
            pd.DataFrame(
                {
                    "Category": [f"{annot}L2_0"],
                    "Coefficient": [1e-8 * z_score],
                    "Coefficient_std_error": [1e-8],
                    "Coefficient_z-score": [z_score],
                    "n_categories": [40],
                    "total_h2": [0.11],
                }
            ).to_csv(trait_dir / f"{annot}.results", sep="\t", index=False)

    results = read_results(results_root, traits=TRAITS, annot2dim=annot2dim)
    attach_enrich_results(embed, ARM, results, factor_selection=fmap)
    write_result(path, embed, provenance=embed.uns["provenance"])

    return {"path": path, "obs": obs_path, "keep": keep}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_obs_decodes_with_an_unfamiliar_index_and_columns(analysis):
    frame = ad.read_h5ad(analysis["obs"]).obs
    assert frame.index.name == "cell_uid"
    assert len(frame) == N_CELLS
    assert set(frame["tissue"].astype(str)) == set(TISSUES)
    # the unmeasured cell has no fraction, not a zero one
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(
            frame["total_reads"].to_numpy() > 0,
            frame["reads_in_peaks"].to_numpy() / frame["total_reads"].to_numpy(),
            np.nan,
        )
    assert np.isnan(fraction[0])


def test_full_result_loads(analysis):
    embed = ad.read_h5ad(analysis["path"])
    arm = embed.uns["enrich"][ARM]
    assert int(arm["factor_selection"]["kept"].sum()) == N_KEPT
    assert set(arm["results"]["trait"]) == set(TRAITS)
    assert len(arm["results"]) == N_KEPT * len(TRAITS)
    assert embed.obsm["X_umap"].shape == (N_CELLS, 2)
    assert "fine_type" in embed.obs.columns
    assert embed.n_obs == N_CELLS


def test_scores_and_aggregation_over_alien_groupings(analysis):
    embed = ad.read_h5ad(analysis["path"])
    arm = embed.uns["enrich"][ARM]
    keep = analysis["keep"]

    primary = arm["results"].loc[arm["results"]["trait"] == "plant_height"]
    assert int(significant(primary).sum()) >= 1

    loadings = directional_loadings(embed, "pos")[keep]
    scores = cs_from_z(loadings, primary, model=ARM, trait="plant_height")
    assert scores.kind is ScoreKind.Z_WEIGHTED
    assert scores.null == 0.0
    assert (scores.values >= 0).all()

    cells = embed.obs
    summary = summarize_by(scores.values, cells, by=["fine_type"], min_cells=10)
    assert len(summary) == len(
        [t for lineage in LINEAGES for t in FINE_TYPES[lineage]]
    )

    means, counts = group_matrix(
        scores.values, cells, index="fine_type", columns="tissue", min_cells=10
    )
    assert means.shape == (8, len(TISSUES))

    blocked = summarize_by(
        scores.values, cells, by=["lineage", "fine_type"], min_cells=10
    )
    order, blocks = block_order(blocked, group="fine_type", block="lineage")
    # `lineage` is a categorical, so blocks follow its DECLARED order rather than an
    # alphabetical one -- re-sorting would throw away the ordering the dtype records.
    assert [name for name, _, _ in blocks] == list(LINEAGES)
    assert len(order) == 8
    # every fine type lands inside its own lineage's span
    for lineage, start, stop in blocks:
        assert set(order[start:stop]) == set(FINE_TYPES[lineage])

    rho2 = eta_squared(
        scores.values.to_numpy(), cells["cultivar"].astype(str).to_numpy()
    )
    assert 0.0 <= rho2 <= 1.0


def test_every_figure_draws_and_saves(analysis, tmp_path):
    embed = ad.read_h5ad(analysis["path"])
    arm = embed.uns["enrich"][ARM]
    keep = analysis["keep"]

    primary = arm["results"].loc[arm["results"]["trait"] == "plant_height"]
    loadings = directional_loadings(embed, "pos")[keep]
    scores = cs_from_z(loadings, primary, model=ARM, trait="plant_height")

    cells = embed.obs.copy()
    cells["embed_1"] = embed.obsm["X_umap"][:, 0]
    cells["embed_2"] = embed.obsm["X_umap"][:, 1]
    cells["score"] = scores.values.reindex(cells.index)

    outdir = tmp_path / "figures"
    written: list = []

    fig, _ = heritability_landscape(arm["results"], trait="plant_height")
    written += save_figure(fig, "01_landscape", outdir)

    fig, _ = trait_concordance(arm["results"], traits=list(TRAITS))
    written += save_figure(fig, "02_concordance", outdir)

    fig, _ = covariate_audit(
        cells, value="score", covariates=["total_reads", "peak_fraction"],
        log_x=["total_reads"],
    )
    written += save_figure(fig, "03_covariates", outdir)

    fig, _ = umap_categorical(cells, "fine_type", x="embed_1", y="embed_2", n=500)
    written += save_figure(fig, "04a_groups", outdir)
    fig, _ = umap_continuous(
        cells, "score", x="embed_1", y="embed_2", n=500, colorbar_label=scores.label
    )
    written += save_figure(fig, "04b_score", outdir)

    stats = box_stats(cells["score"], cells["fine_type"].astype(str), min_n=10)
    fig, _ = score_by_group(
        stats, group_label="fine_type", value_label=scores.label, null=scores.null,
        highlight=["vasc_1"],
    )
    written += save_figure(fig, "05_by_group", outdir)

    means, counts = group_matrix(
        scores.values, cells, index="fine_type", columns="tissue",
        min_cells=10, column_order=list(TISSUES),
    )
    fig, _ = grouped_landscape(
        means, counts, row_label="fine_type", column_label="tissue",
        value_label=scores.label,
    )
    written += save_figure(fig, "06_landscape", outdir)

    assert len(written) == 14  # seven figures, pdf + png each
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_highlight_names_a_category_from_this_dataset(analysis):
    """The highlighted category is the caller's claim, so an alien name must work and
    an unknown one must be refused."""
    from scads_drvi.pl.color import categorical_palette

    palette = categorical_palette(list(FINE_TYPES["vasculature"]), highlight="vasc_2")
    assert palette["vasc_2"] == "#D55E00"
    with pytest.raises(KeyError, match="cannot highlight"):
        categorical_palette(list(TISSUES), highlight="Immune")
