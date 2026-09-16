"""One-time data provisioning for pbmc.ipynb: 10x PBMC download + QC/cell-typing,
the RA (Ishigaki et al. 2022) GWAS sumstats, and the 1000G/baselineLD-v2.2 reference
panel. Every step checks its own output first and skips it if already present, so
re-running this script (from the notebook or the command line) only fills gaps.

Run directly: python prepare_data.py
"""

import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TENX_BASE = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k"

# GWAS Catalog GCST90132222 -- Ishigaki et al. 2022 RA, EUR meta-analysis (97,173
# European-ancestry samples), the same accession this tutorial's own text describes.
SUMSTATS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
    "GCST90132001-GCST90133000/GCST90132222/GCST90132222_buildGRCh37.tsv.gz"
)
SUMSTATS_MD5 = "dffee6d90b36d03f129d1d49d305875b"
SUMSTATS_N = 97_173  # total EUR N (GWAS Catalog metadata) -- no per-SNP N column

# A sibling project's already-downloaded 1000G EUR hg38 + baselineLD v2.2 reference --
# personal to this cluster/user, copied once into this tutorial's own data/ so the
# notebook stays self-contained after that.
CAPSULE_REF = Path(
    "/allen/programs/celltypes/workgroups/hct/mike.cuoco/capsule-5254633/data/"
    "seaad_shared_references/broad_alkesgroup/ld_score_regression/grch38_baselineLD_v2.2"
)


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`. A real browser User-Agent -- some CDNs (10x's included)
    403 the default `Python-urllib/...` string as a blanket anti-scraping heuristic,
    even for public, unauthenticated files like this one.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)


def download_10x(data_dir: Path = DATA) -> Path:
    """10x multiome PBMC h5 -- paired RNA+ATAC, real genomic peak coordinates."""
    data_dir.mkdir(exist_ok=True)
    matrix_path = data_dir / "pbmc_multiome_filtered_feature_bc_matrix.h5"
    if not matrix_path.exists():
        print(f"downloading {matrix_path.name}...")
        _download(
            f"{TENX_BASE}/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5",
            matrix_path,
        )
    print(matrix_path, matrix_path.stat().st_size / 1e6, "MB")
    return matrix_path


def prepare_atac_rna(matrix_path: Path, data_dir: Path = DATA) -> tuple[Path, Path]:
    """QC/filter both modalities, keep the most-accessible peaks, call real cell
    types from real RNA markers, and carry that call over to the ATAC modality by
    barcode. Writes data/pbmc_rna.h5ad and data/pbmc_atac.h5ad; skips recompute if
    both already exist.
    """
    rna_path, atac_path = data_dir / "pbmc_rna.h5ad", data_dir / "pbmc_atac.h5ad"
    if rna_path.exists() and atac_path.exists():
        print(f"already prepared: {rna_path}, {atac_path}")
        return rna_path, atac_path

    import scanpy as sc

    raw = sc.read_10x_h5(matrix_path, gex_only=False)
    raw.var_names_make_unique()

    rna = raw[:, raw.var["feature_types"] == "Gene Expression"].copy()
    atac = raw[:, raw.var["feature_types"] == "Peaks"].copy()

    # canonical chr:start-end peaks on primary chromosomes only
    PEAK_RE = re.compile(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y):\d+-\d+$")
    atac = atac[:, atac.var_names.str.match(PEAK_RE)].copy()

    sc.pp.calculate_qc_metrics(rna, percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    keep_cells = (rna.obs["total_counts"] >= 500) & (atac.obs["total_counts"] >= 1000)
    rna, atac = rna[keep_cells].copy(), atac[keep_cells].copy()

    sc.pp.filter_genes(rna, min_cells=10)
    sc.pp.filter_genes(atac, min_cells=10)

    # most-accessible peaks only -- keeps a CPU-scale DRVI fit tractable
    N_PEAKS = 20_000
    top_peaks = atac.var["total_counts"].sort_values(ascending=False).index[:N_PEAKS]
    atac = atac[:, atac.var_names.isin(top_peaks)].copy()

    print(f"rna {rna.shape}, atac {atac.shape}")

    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000)
    sc.pp.pca(rna, n_comps=30, mask_var="highly_variable")
    sc.pp.neighbors(rna, n_neighbors=15)
    sc.tl.umap(rna, min_dist=0.3, spread=1.0)
    sc.tl.leiden(rna, resolution=0.5, flavor="leidenalg", n_iterations=2)

    MARKERS = {
        "CD4 T": ["IL7R", "CD3D", "CD3E", "CD4"],
        "CD8 T": ["CD3D", "CD3E", "CD8A", "CD8B"],
        "NK": ["GNLY", "NKG7", "KLRD1"],
        "B": ["MS4A1", "CD79A", "CD79B"],
        "Monocyte": ["CD14", "LYZ", "FCN1"],
        "DC": ["FCER1A", "CST3", "CLEC10A"],
        "Platelet": ["PPBP", "PF4"],
    }
    score_cols = []
    for name, genes in MARKERS.items():
        present = [g for g in genes if g in rna.var_names]
        if present:
            sc.tl.score_genes(rna, present, score_name=f"score_{name}")
            score_cols.append(f"score_{name}")

    cluster_scores = rna.obs.groupby("leiden")[score_cols].mean()
    cluster_label = cluster_scores.idxmax(axis=1).str.replace("score_", "", regex=False)
    rna.obs["cell_type"] = rna.obs["leiden"].map(cluster_label).astype("category")

    atac.obs["cell_type"] = rna.obs.loc[atac.obs_names, "cell_type"].values
    atac.obs["n_genes_rna"] = rna.obs.loc[atac.obs_names, "n_genes_by_counts"].values
    atac.obs["n_counts_rna"] = rna.obs.loc[atac.obs_names, "total_counts"].values

    print(rna.obs["cell_type"].value_counts())

    rna.write_h5ad(rna_path)
    atac.write_h5ad(atac_path)
    return rna_path, atac_path


def provision_ldsc_reference(data_dir: Path = DATA) -> Path:
    """Copy the 1000G EUR hg38 + baselineLD v2.2 reference panel (~3.3G) into this
    tutorial's own data/ once, so the notebook doesn't depend on another project's
    path. Uses rsync so an interrupted copy resumes instead of restarting.
    """
    dest = data_dir / "ldsc_reference" / CAPSULE_REF.name
    if dest.exists() and any(dest.iterdir()):
        print(f"already provisioned: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"copying reference panel from {CAPSULE_REF} to {dest}...")
    subprocess.run(["rsync", "-a", f"{CAPSULE_REF}/", f"{dest}/"], check=True)
    return dest


def download_and_munge_sumstats(data_dir: Path = DATA, ref_dir: Path | None = None) -> Path:
    """Download the real RA (Ishigaki et al. 2022) EUR sumstats from the GWAS
    Catalog, verify their checksum, and munge them into LDSC-ready format.
    """
    if ref_dir is None:
        ref_dir = data_dir / "ldsc_reference" / CAPSULE_REF.name
    sumstats_dir = data_dir / "sumstats"
    sumstats_dir.mkdir(parents=True, exist_ok=True)

    raw_path = sumstats_dir / "GCST90132222_buildGRCh37.tsv.gz"
    if not raw_path.exists():
        print(f"downloading {raw_path.name}...")
        _download(SUMSTATS_URL, raw_path)
    digest = hashlib.md5(raw_path.read_bytes()).hexdigest()
    if digest != SUMSTATS_MD5:
        raise ValueError(f"{raw_path} md5 {digest} != expected {SUMSTATS_MD5}")

    munged_path = sumstats_dir / "ra_ishigaki2022_eur.sumstats.gz"
    if munged_path.exists():
        print(f"already munged: {munged_path}")
        return munged_path

    from scads_drvi.enrich.binary import run_ldsc

    run_ldsc("munge-sumstats", {
        "sumstats": str(raw_path),
        "snp": "variant_id", "a1": "effect_allele", "a2": "other_allele",
        "p": "p_value", "signed_sumstats": "beta,0", "N": SUMSTATS_N,
        "merge_alleles": str(ref_dir / "w_hm3.snplist"),
        "out": str(sumstats_dir / "ra_ishigaki2022_eur"),
    })
    if not munged_path.exists():
        raise FileNotFoundError(f"munge-sumstats did not produce {munged_path}")
    return munged_path


def main() -> None:
    matrix_path = download_10x()
    prepare_atac_rna(matrix_path)
    ref_dir = provision_ldsc_reference()
    download_and_munge_sumstats(ref_dir=ref_dir)


if __name__ == "__main__":
    main()
