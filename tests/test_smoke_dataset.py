"""The loaders against a real, public dataset written by anndata.

`test_portability.py` proves the package assumes nothing about a *vocabulary*, using an
h5ad hand-built with h5py. This proves it assumes nothing wrong about an *encoding*, using
a file anndata actually wrote -- which is the only way the backed-CSR reader in `io.h5ad`
is exercised at all: `read_rows_csr`, `h5ad_shape`, `h5ad_var_names` and `keep_rows` have
no other caller in this repository.

Skips unless the dataset is cached or downloading is enabled; see `tests/smoke_data.py` for
what is real here and what is fabricated.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("anndata")

from smoke_data import (  # noqa: E402
    EXPECTED_N_OBS,
    EXPECTED_N_VARS,
    EXPECTED_NNZ,
    LAYOUT_VERSION,
    SmokeUnavailable,
    ensure_matrix,
    synthesize_run,
)

from scads_drvi.io.artifacts import (  # noqa: E402
    cell_metadata,
    load_interpretation,
    read_loadings,
    read_obs,
)
from scads_drvi.io.contract import check_contract  # noqa: E402
from scads_drvi.io.h5ad import (  # noqa: E402
    h5ad_shape,
    h5ad_var_names,
    keep_rows,
    read_rows_csr,
)
from scads_drvi.io.peaks import PEAK_RE, normalize_peak_name  # noqa: E402
from scads_drvi.scores.aggregate import (  # noqa: E402
    block_order,
    eta_squared,
    group_matrix,
    summarize_by,
)
from scads_drvi.scores.cell import cs_from_z  # noqa: E402

pytestmark = pytest.mark.smoke

#: Strata as `smoke_data` declares them -- deliberately not alphabetical, so an
#: implementation that sorts instead of honouring the dtype fails visibly.
FRIP_LEVELS = ["low", "mid", "high"]
SIGNAL_LEVELS = [
    "low.distal",
    "low.promoter",
    "mid.distal",
    "mid.promoter",
    "high.distal",
    "high.promoter",
]
DUPLICATE_LEVELS = ["weak", "moderate", "strong"]

GROUPINGS = ["frip_stratum", "signal_stratum", "duplicate_stratum"]
DERIVED = {
    "frip": ("peak_region_fragments", "n_fragment"),
    "tss_fraction": ("TSS_fragments", "n_fragment"),
}


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """The real dataset, plus a fabricated run tree over its real cells."""
    allow_download = os.environ.get("SCADS_DRVI_SMOKE_DOWNLOAD") == "1"
    required = os.environ.get("SCADS_DRVI_SMOKE_REQUIRED") == "1"
    try:
        path, manifest = ensure_matrix(allow_download=allow_download)
    except SmokeUnavailable as exc:
        # Required is what stops a CI job that has quietly lost network from reporting a
        # row of green skips and calling it a pass.
        if required:
            pytest.fail(f"smoke dataset required but unavailable: {exc}")
        pytest.skip(f"smoke dataset unavailable: {exc}")

    run = synthesize_run(tmp_path_factory.mktemp("synthetic_run"), path)
    return {"path": path, "manifest": manifest, **run}


@pytest.fixture(scope="module")
def cells(smoke):
    return cell_metadata(smoke["path"], derived=DERIVED)


def test_matrix_is_cells_by_peaks_in_the_encoding_the_reader_assumes(smoke):
    """The canary: `io.h5ad` reads X with raw h5py, so the writer's layout is a contract."""
    import h5py

    assert h5ad_shape(str(smoke["path"])) == (EXPECTED_N_OBS, EXPECTED_N_VARS)

    with h5py.File(smoke["path"], "r") as handle:
        matrix = handle["X"]
        assert {"data", "indices", "indptr"} <= set(matrix.keys())
        assert tuple(int(v) for v in matrix.attrs["shape"]) == (
            EXPECTED_N_OBS,
            EXPECTED_N_VARS,
        )
        assert matrix.attrs["encoding-type"] == "csr_matrix"
        assert matrix["data"].shape[0] == EXPECTED_NNZ
        # indptr indexes cells, which is the whole orientation question in one number
        assert matrix["indptr"].shape[0] == EXPECTED_N_OBS + 1


def test_var_names_are_canonical_peaks(smoke):
    names = h5ad_var_names(str(smoke["path"]))
    assert len(names) == EXPECTED_N_VARS
    assert len(set(names)) == EXPECTED_N_VARS

    bad = [n for n in names if not PEAK_RE.match(n)]
    assert not bad, f"{len(bad)} non-canonical peak names, e.g. {bad[:3]}"
    # already canonical, so normalising is the identity -- the round trip nothing else checks
    assert [normalize_peak_name(n) for n in names[:500]] == list(names[:500])


def test_row_sums_track_the_reported_peak_signal(smoke, cells):
    """Orientation. A transposed or misjoined matrix cannot produce this correlation."""
    import anndata as ad

    adata = ad.read_h5ad(smoke["path"])
    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()

    signal = cells["peak_region_cutsites"].to_numpy(dtype=float)
    assert np.corrcoef(row_sums, signal)[0, 1] > 0.999
    # and it is a real discriminator, not something any column would satisfy
    depth = cells["n_fragment"].to_numpy(dtype=float)
    assert np.corrcoef(row_sums, depth)[0, 1] < 0.9


def test_read_rows_csr_is_invariant_to_max_gap(smoke):
    """`max_gap` only chooses how much to over-read; it must never change the answer."""
    import anndata as ad

    adata = ad.read_h5ad(smoke["path"])
    rng = np.random.default_rng(0)
    rows = rng.choice(adata.shape[0], size=37, replace=False)  # scattered and unsorted
    assert not np.all(rows[:-1] <= rows[1:]), "the un-sort path needs unsorted input"

    want = adata.X[rows].toarray()
    for max_gap in (0, None, 10**9):
        got = read_rows_csr(str(smoke["path"]), rows, adata.shape[1], max_gap=max_gap)
        # equality, in the CALLER's row order -- the inverse permutation at the end of
        # read_rows_csr is otherwise untested
        assert np.array_equal(got.toarray(), want), f"max_gap={max_gap}"


def test_keep_rows_gates_on_the_real_depth_distribution(smoke, cells):
    depth = cells["n_fragment"].to_numpy()
    threshold = int(np.median(depth))

    rows, info = keep_rows(str(smoke["path"]), threshold)
    assert info["depth_col"] == "n_fragment"
    assert info["n_cells_available"] == EXPECTED_N_OBS
    assert info["n_cells_kept"] == int((depth >= threshold).sum())
    assert info["n_cells_kept"] + info["n_cells_excluded"] == EXPECTED_N_OBS
    assert np.array_equal(rows, np.flatnonzero(depth >= threshold))

    everything, none_info = keep_rows(str(smoke["path"]), 0)
    assert none_info is None
    assert len(everything) == EXPECTED_N_OBS


def test_obs_written_by_anndata_decodes(smoke, cells):
    frame = read_obs(smoke["path"], ["n_fragment", "frip_stratum"])
    assert frame.index.name == "barcode"
    assert len(frame) == EXPECTED_N_OBS
    assert str(frame["n_fragment"].dtype).startswith("int")

    # The declared, non-alphabetical category ORDER survives the round trip, and that order
    # is what block_order consumes -- so it is the half that carries weight.
    #
    # The `ordered` FLAG does not survive: smoke_data writes these with ordered=True, and
    # read_obs rebuilds them with pd.Categorical.from_codes, which defaults to ordered=False
    # and never consults the `ordered` attribute anndata wrote. Nothing in the package
    # compares strata with < or >, so this costs nothing today; it is asserted here as the
    # observed behaviour rather than the intended one, so a future fix shows up as a failing
    # test instead of a silent semantic change.
    assert list(cells["frip_stratum"].cat.categories) == FRIP_LEVELS
    assert list(cells["signal_stratum"].cat.categories) == SIGNAL_LEVELS
    assert list(cells["duplicate_stratum"].cat.categories) == DUPLICATE_LEVELS
    assert FRIP_LEVELS != sorted(FRIP_LEVELS)
    assert cells["frip_stratum"].cat.ordered is False

    # Ratios are derived on read, never stored, so this is cell_metadata doing real division
    # over real counts rather than reading back something precomputed.
    for name in DERIVED:
        values = cells[name].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
    assert cells["frip"].std() > 0.05, "FRiP should vary across real cells"

    # Every stratum is populated, and nested strata sit strictly inside their parent.
    assert set(cells["signal_stratum"].astype(str)) == set(SIGNAL_LEVELS)
    for coarse, fine in zip(
        cells["frip_stratum"].astype(str), cells["signal_stratum"].astype(str), strict=True
    ):
        assert fine.split(".")[0] == coarse


def test_contract_check_passes_on_the_synthesized_fit(smoke):
    project = smoke["project"]
    report = check_contract(project.fit_dir(), log_fn=lambda _msg: None)
    assert report["n_cells"] == EXPECTED_N_OBS
    assert report["n_latent"] == 12
    # topic_factors.tsv is indexed by the real peaks, so this is check_contract's PEAK_RE
    # branch running over a real vocabulary
    assert report["topic_factors"] is True
    assert report["n_features"] == EXPECTED_N_VARS


def test_aggregation_over_real_qc_strata(smoke, cells):
    project, arm = smoke["project"], smoke["arm"]
    trait = smoke["traits"][0]

    interp = load_interpretation(
        project,
        arm,
        obs_path=smoke["path"],
        obs_columns=GROUPINGS + ["n_fragment"],
        derived=DERIVED,
        umap_path=smoke["coords"],
    )
    assert interp.n_cells == EXPECTED_N_OBS
    assert interp.labels.n_kept == 9

    contract = project.contract(arm)
    loadings = read_loadings(
        contract["loadings"],
        npz=contract["loadings_npz"],
        dims=list(interp.labels.kept_dims),
    )
    scores = cs_from_z(
        loadings, interp.for_trait(trait), model=arm, trait=trait, labels=interp.labels
    )
    assert (scores.values >= 0).all()

    frame = interp.cells
    summary = summarize_by(scores.values, frame, by=["signal_stratum"], min_cells=20)
    assert len(summary) == len(SIGNAL_LEVELS)

    means, counts = group_matrix(
        scores.values,
        frame,
        index="signal_stratum",
        columns="duplicate_stratum",
        min_cells=10,
    )
    assert means.shape == (len(SIGNAL_LEVELS), len(DUPLICATE_LEVELS))
    assert counts.to_numpy().sum() == EXPECTED_N_OBS

    blocked = summarize_by(
        scores.values, frame, by=["frip_stratum", "signal_stratum"], min_cells=20
    )
    order, blocks = block_order(blocked, group="signal_stratum", block="frip_stratum")
    # declared order, not alphabetical -- 'high' sorts first alphabetically and must not
    assert [name for name, _, _ in blocks] == FRIP_LEVELS
    for name, start, stop in blocks:
        assert all(group.startswith(f"{name}.") for group in order[start:stop])

    explained = eta_squared(
        scores.values.to_numpy(), frame["frip_stratum"].astype(str).to_numpy()
    )
    assert 0.0 <= explained <= 1.0


def test_a_figure_survives_a_real_heavy_tailed_covariate(smoke, tmp_path):
    """Real depth spans two orders of magnitude; the synthetic fixtures are uniform."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scads_drvi.viz.enrichment import covariate_audit
    from scads_drvi.viz.save import save_figure

    interp = load_interpretation(
        smoke["project"],
        smoke["arm"],
        obs_path=smoke["path"],
        obs_columns=GROUPINGS + ["n_fragment"],
        derived=DERIVED,
        umap_path=smoke["coords"],
    )
    frame = interp.cells.copy()
    depth = frame["n_fragment"].to_numpy(dtype=float)
    assert depth.max() / max(depth.min(), 1) > 20, "expected a heavy tail to plot"

    trait = smoke["traits"][0]
    contract = smoke["project"].contract(smoke["arm"])
    loadings = read_loadings(
        contract["loadings"],
        npz=contract["loadings_npz"],
        dims=list(interp.labels.kept_dims),
    )
    scores = cs_from_z(loadings, interp.for_trait(trait), model=smoke["arm"], trait=trait)
    frame["score"] = scores.values.reindex(frame.index)

    try:
        figure, _axes = covariate_audit(
            frame, value="score", covariates=["n_fragment", "frip"], log_x=["n_fragment"]
        )
        written = save_figure(figure, "covariate_audit", tmp_path)
    finally:
        plt.close("all")
    assert written
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_provenance_is_recorded(smoke):
    manifest = smoke["manifest"]
    assert manifest["layout_version"] == LAYOUT_VERSION
    assert (manifest["n_obs"], manifest["n_vars"]) == (EXPECTED_N_OBS, EXPECTED_N_VARS)
    assert manifest["nnz"] == EXPECTED_NNZ
    for entry in manifest["files"].values():
        assert entry["url"].startswith("https://")
        assert len(entry["sha256"]) == 64
        assert entry["n_bytes"] > 0

    # The fabricated half says so in the artifact a reader would open first.
    meta = json.loads((smoke["project"].fit_dir() / "fit.meta.json").read_text())
    assert meta["synthetic"] is True
    assert meta["generator"] == "tests/smoke_data.py"
    assert smoke["fit"].startswith("synthetic")
    assert smoke["arm"].startswith("synthetic")
    assert all(trait.startswith("synthetic") for trait in smoke["traits"])


def test_it_refuses_to_download_when_downloading_is_off(tmp_path):
    """The default for anyone running `pytest`: no network, a named reason."""
    with pytest.raises(SmokeUnavailable) as caught:
        ensure_matrix(cache=tmp_path, allow_download=False)
    message = str(caught.value)
    assert "SCADS_DRVI_SMOKE_DOWNLOAD" in message
    assert "https://" in message
