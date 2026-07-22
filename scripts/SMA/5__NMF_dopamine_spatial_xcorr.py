#%% Load data
# conda env: nichecompass_liana

import os
for thread_env_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[thread_env_var] = "1"

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env", override=True)

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

#%% Load joint data
# Mirrors the save path at the end of 3__spatial_meta_modelling.py:
#   OUTPUT_DIR / "spatialjepa_models" / RUN_ID / "joint_adata.h5ad"
# RUN_ID is the shared section prefix for multi-section runs (e.g. "V11T17-102")
# or the single sample ID for single-section runs.
RUN_ID = "V11L12-109_B1"           # <-- set to match the run in script 3
SECTION_KEY = "section"         # obs column written by script 3

OUTPUT_DIR = Path(os.getenv("OUTPATH"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / RUN_ID
PROJECTION_DIR = OUTPUT_DIR / "spatialjepa_projection" / RUN_ID

# NOTE: joint_adata here is scoped to the NMF/bivariate-Moran's-I analysis below only.
# The MOFA+ section further down loads its own separate MuData objects from
# spatial_target_trimodal_mudata_path / multiome_target_trimodal_mudata_path and never
# references joint_adata, so this toggle cannot affect the MOFA+ results.
USE_NO_PANEL_JOINT_ADATA = True   # <-- flip to False to use panel-restricted joint_adata.h5ad
joint_adata_path = MODEL_DIR / (
    "joint_adata_no_panel.h5ad" if USE_NO_PANEL_JOINT_ADATA else "joint_adata.h5ad"
)
if not joint_adata_path.exists():
    if USE_NO_PANEL_JOINT_ADATA:
        raise FileNotFoundError(
            f"Missing {joint_adata_path}. Run 3__spatial_meta_modelling.py for RUN_ID={RUN_ID!r} "
            "to create joint_adata_no_panel.h5ad, or set USE_NO_PANEL_JOINT_ADATA=False "
            "to use the panel-restricted joint_adata.h5ad."
        )
    raise FileNotFoundError(
        f"Missing {joint_adata_path}. Run 3__spatial_meta_modelling.py for RUN_ID={RUN_ID!r} "
        "to create the panel-restricted joint_adata.h5ad."
    )

joint_adata = sc.read_h5ad(joint_adata_path)
print(
    f"[INFO] loaded {joint_adata_path.name}: {joint_adata.n_obs} spots, "
    f"{joint_adata.n_vars} features ({joint_adata.var['type'].eq('ST').sum()} ST)"
)

def as_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

#%% Run NMF on spatial Visium data

from sklearn.decomposition import NMF
from tqdm import tqdm

def zscore(v):
    v = np.asarray(v, dtype=float).ravel()
    return (v - v.mean()) / v.std()

def fit_nmf_spatial(x, n_components, spatial, **nmf_kwargs):
    """Fit NMF and wrap the factor scores in a spatially-annotated AnnData."""
    nmf = NMF(n_components=n_components, **nmf_kwargs)
    x_nmf = nmf.fit_transform(x)
    adata = sc.AnnData(X=x_nmf, obsm={"spatial": spatial})
    return adata, nmf

def bivariate_morans_i(X, lag_y, n, S0):
    """Bivariate Moran's I of each column of X against a precomputed spatial lag."""
    return [
        (n / S0) * (zscore(X[:, k]) @ lag_y) / (zscore(X[:, k]) @ zscore(X[:, k]))
        for k in range(X.shape[1])
    ]

# Visium hex lattice: 6 neighbours; library_key blocks cross-section edges for multi-section runs
sq.gr.spatial_neighbors(joint_adata, coord_type="grid", n_neighs=6, library_key=SECTION_KEY)
W = joint_adata.obsp['spatial_connectivities']
S0 = W.sum()
n = joint_adata.n_obs

st_features = joint_adata.var_names[joint_adata.var['type'].eq('ST').to_numpy()]
if USE_NO_PANEL_JOINT_ADATA:
    # assemble_section() in 3__spatial_meta_modelling.py prefers SCT-transformed h5mu
    # files when available, so for joint_adata_no_panel.h5ad the 'normalized' layer is
    # SCT_data-derived (already log-normalized) rather than raw-count-derived. Use
    # SCT_counts (SCT-corrected, non-negative counts) instead, since KL-NMF expects
    # non-negative, non-log inputs.
    # NOTE: the best-performing solution found so far (max bivariate Moran's I ~0.50)
    # was still the raw-counts + total-counts-normalization feature ('normalized'),
    # built before SCT was introduced into assemble_section(). SCT_counts is the closest
    # available substitute now that 'normalized' is SCT-contaminated for this run.
    x_visium = as_dense(joint_adata[:, st_features].layers['SCT_counts'])
else:
    x_visium = as_dense(joint_adata[:, st_features].layers['normalized'])
x_dopamine = as_dense(joint_adata[:, 'msi:Dopamine'].X).squeeze()
zy = zscore(x_dopamine)                         # the "lagged" variable
lag_y = W @ zy                                   # spatial lag of dopamine

# Poisson NMF with KL loss
from joblib import dump, load

nmf_kwargs = dict(
    init='nndsvda',
    solver='mu',
    beta_loss='kullback-leibler',
    max_iter=1000,   # mu converges slower than cd's default 200
    tol=1e-4,
    random_state=0,
)

RESUME_NMF_IF_AVAILABLE = True
NMF_CACHE_STEM = (
    "nmf_dopamine_spatial_xcorr"
    + ("_no_panel" if USE_NO_PANEL_JOINT_ADATA else "_panel")
)


def nmf_cache_paths(model_dir=PROJECTION_DIR, cache_stem=NMF_CACHE_STEM):
    """Paths for the trained NMF model, factor AnnData, and sweep metrics."""
    model_dir = Path(model_dir)
    return {
        "model": model_dir / f"{cache_stem}_model.joblib",
        "adata": model_dir / f"{cache_stem}_factors.h5ad",
        "metrics": model_dir / f"{cache_stem}_metrics.npz",
    }


def save_trained_nmf_for_downstream(
    nmf,
    nmf_adata,
    *,
    sweep_n_components,
    max_bivariate_moran_Is,
    bivariate_moran_I,
    cache_paths=None,
):
    """Persist fitted NMF outputs needed to resume the downstream plots."""
    if cache_paths is None:
        cache_paths = nmf_cache_paths()
    for path in cache_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    dump(nmf, cache_paths["model"])
    nmf_adata.write_h5ad(cache_paths["adata"])
    np.savez(
        cache_paths["metrics"],
        sweep_n_components=np.asarray(sweep_n_components),
        max_bivariate_moran_Is=np.asarray(max_bivariate_moran_Is),
        bivariate_moran_I=np.asarray(bivariate_moran_I),
        best_n_components=np.asarray(nmf.components_.shape[0]),
    )
    print(f"Saved trained NMF cache to {cache_paths['model'].parent}")


def load_trained_nmf_for_downstream(*, st_features, cache_paths=None):
    """Load fitted NMF outputs and recreate variables used by downstream plots."""
    if cache_paths is None:
        cache_paths = nmf_cache_paths()
    missing = [str(path) for path in cache_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trained NMF cache files:\n" + "\n".join(missing))

    nmf = load(cache_paths["model"])
    nmf_adata = sc.read_h5ad(cache_paths["adata"])
    with np.load(cache_paths["metrics"]) as metrics:
        sweep_n_components = metrics["sweep_n_components"]
        max_bivariate_moran_Is = metrics["max_bivariate_moran_Is"]
        bivariate_moran_I = metrics["bivariate_moran_I"]
        best_n_components = int(metrics["best_n_components"])

    H = nmf.components_
    nmf_components = list(nmf_adata.var_names)
    best_bivariate_moran_idx = int(np.argmax(bivariate_moran_I))
    best_bivariate_moran_component = nmf_components[best_bivariate_moran_idx]
    best_bivariate_moran_score = bivariate_moran_I[best_bivariate_moran_idx]
    bivariate_moran_I_df = (
        pd.DataFrame(bivariate_moran_I, index=nmf_components)
        .sort_values(0, ascending=True)
        .rename(columns={0: "bivariate_moran_I"})
    )
    best_nmf_component = pd.Series(nmf.components_[best_bivariate_moran_idx], index=st_features)
    best_nmf_component.index = best_nmf_component.index.str.split(':').str[1]
    print(f"Loaded trained NMF cache from {cache_paths['model'].parent}")
    return (
        nmf_adata,
        nmf,
        H,
        sweep_n_components,
        max_bivariate_moran_Is,
        best_n_components,
        bivariate_moran_I,
        nmf_components,
        best_bivariate_moran_idx,
        best_bivariate_moran_component,
        best_bivariate_moran_score,
        bivariate_moran_I_df,
        best_nmf_component,
    )


sweep_n_components = np.arange(1, 30)
try:
    if not RESUME_NMF_IF_AVAILABLE:
        raise FileNotFoundError("NMF resume disabled.")
    (
        nmf_adata,
        nmf,
        H,
        sweep_n_components,
        max_bivariate_moran_Is,
        best_n_components,
        bivariate_moran_I,
        nmf_components,
        best_bivariate_moran_idx,
        best_bivariate_moran_component,
        best_bivariate_moran_score,
        bivariate_moran_I_df,
        best_nmf_component,
    ) = load_trained_nmf_for_downstream(st_features=st_features)
except FileNotFoundError as exc:
    print(f"{exc}\nTraining NMF from scratch.")
    max_bivariate_moran_Is = []
    for n_components in tqdm(sweep_n_components):
        nmf_adata, _ = fit_nmf_spatial(x_visium, n_components, joint_adata.obsm["spatial"], **nmf_kwargs)
        bivariate_moran_I = bivariate_morans_i(nmf_adata.X, lag_y, n, S0)
        max_bivariate_moran_Is.append(np.max(bivariate_moran_I))

    best_n_components = sweep_n_components[np.argmax(max_bivariate_moran_Is)]
    print(f"Best number of NMF components: {best_n_components}")

    nmf_adata, nmf = fit_nmf_spatial(x_visium, best_n_components, joint_adata.obsm["spatial"], **nmf_kwargs)
    H = nmf.components_
    bivariate_moran_I = bivariate_morans_i(nmf_adata.X, lag_y, n, S0)
    nmf_components = list(nmf_adata.var_names)
    best_bivariate_moran_idx = int(np.argmax(bivariate_moran_I))
    best_bivariate_moran_component = nmf_components[best_bivariate_moran_idx]
    best_bivariate_moran_score = bivariate_moran_I[best_bivariate_moran_idx]
    bivariate_moran_I_df = (
        pd.DataFrame(bivariate_moran_I, index=nmf_components)
        .sort_values(0, ascending=True)
        .rename(columns={0: "bivariate_moran_I"})
    )
    best_nmf_component = pd.Series(nmf.components_[best_bivariate_moran_idx], index=st_features)
    best_nmf_component.index = best_nmf_component.index.str.split(':').str[1]
    save_trained_nmf_for_downstream(
        nmf,
        nmf_adata,
        sweep_n_components=sweep_n_components,
        max_bivariate_moran_Is=max_bivariate_moran_Is,
        bivariate_moran_I=bivariate_moran_I,
    )

plt.figure(figsize=(10, 5))
plt.plot(sweep_n_components, max_bivariate_moran_Is, marker='o')
plt.xlabel("Number of NMF components")
plt.ylabel("Max bivariate Moran's I")
plt.title("Max bivariate Moran's I vs. Number of NMF components")

print(f"Best number of NMF components: {best_n_components}")

# Plot NMF components as subfigures in the same Scanpy figure.
nmf_component_titles = [
    f"{comp} (biv. I = {bivariate_moran_I[k]:.3f})"
    for k, comp in enumerate(nmf_components)
]
embedding_axes = sc.pl.embedding(
    nmf_adata,
    color=nmf_components,
    basis="spatial",
    title=nmf_component_titles,
    show=False,
)
embedding_fig = np.ravel(embedding_axes)[0].figure
embedding_fig.suptitle(
    f"Highest bivariate Moran's I: {best_bivariate_moran_component} "
    f"({best_bivariate_moran_score:.3f})",
    fontsize=48,
    fontweight="bold",
    y=0.995,
)
embedding_fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
# %% barplot of bivariate Moran's I

bivariate_moran_I_df.plot(kind='barh')
plt.title('Bivariate Moran\'s I of NMF components against dopamine')
plt.xlabel('Bivariate Moran\'s I')
plt.ylabel('NMF component')
plt.show()

best_nmf_component.sort_values(ascending=True).tail(10).plot(kind='barh')
plt.title('Highest gene loadings on dopamine-correlated NMF component')
plt.xlabel('Gene loading')
plt.ylabel('Gene')
plt.show()

# %% MOFA+ analysis of spatial and multiome targets
import muon as mu
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

spatial_target_trimodal_mudata_path = os.path.join(
    os.getenv("OUTPATH"),
    "spatialjepa_projection",
    "V11L12-109_B1",
    "spatial_target_rna_atac_msi.h5mu",
)

multiome_target_trimodal_mudata_path = os.path.join(
    os.getenv("OUTPATH"),
    "spatialjepa_projection",
    "V11L12-109_B1",
    "multiome_target_rna_atac_msi.h5mu",
)

DEFAULT_MOFA_MODALITIES = ("rna", "atac", "msi_student")


def finite_variable_feature_mask(adata, initial_mask, label):
    """Return a full-length mask keeping finite, non-constant selected features."""
    initial_mask = np.asarray(initial_mask, dtype=bool)
    selected_idx = np.flatnonzero(initial_mask)
    X = adata[:, initial_mask].X

    if sp.issparse(X):
        X = X.tocsc(copy=True)
        finite = np.ones(X.shape[1], dtype=bool)
        if X.data.size:
            bad_data = ~np.isfinite(X.data)
            if np.any(bad_data):
                bad_cols = np.repeat(np.arange(X.shape[1]), np.diff(X.indptr))[bad_data]
                finite[np.unique(bad_cols)] = False
                X.data[bad_data] = 0
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq_mean = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = sq_mean - mean ** 2
    else:
        X = np.asarray(X)
        finite = np.isfinite(X).all(axis=0)
        var = np.nanvar(X, axis=0)

    keep_selected = finite & (var > 0)
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected_idx[keep_selected]] = True
    dropped = int(initial_mask.sum() - mask.sum())
    if dropped:
        print(f"MOFA {label}: dropped {dropped} selected features with non-finite or zero variance values.")
    return mask


def mofa_outfile_from_mudata_path(trimodal_mudata_path):
    """Derive MOFA HDF5 path from a trimodal .h5mu path in the same directory."""
    path = Path(trimodal_mudata_path)
    return str(path.with_name(path.stem.replace("_rna_atac_msi", "_mofa_model") + ".hdf5"))


def close_h5py_handles_for_path(path):
    """Close lingering read/write h5py.File handles on `path`.

    mofapy2 and mofax often leave HDF5 files open after training or loading.
    Re-running MOFA then fails at save with:
    OSError: unable to truncate a file which is already open
    """
    import gc
    import h5py

    path = str(path)
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File) and obj.id.valid and obj.filename == path:
                obj.close()
        except Exception:
            pass


def peaks_to_bed_df(names):
    """Parse 'chrN:start-end' peak/region names into a BED-style DataFrame.

    Keeps the original name in a 'name' column so overlaps can be mapped back.
    Rows that don't match the pattern (malformed coords) are dropped.
    """
    names = pd.Series(np.asarray(names), dtype=str)
    coords = names.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    coords["name"] = names.to_numpy()
    coords = coords.dropna(subset=["chrom", "start", "end"])
    coords["start"] = coords["start"].astype(int)
    coords["end"] = coords["end"].astype(int)
    return coords[["chrom", "start", "end", "name"]]


def atac_peaks_overlapping_regions(atac_var_names, regions):
    """Return a boolean mask over atac_var_names for peaks overlapping any region.

    ATAC peaks and cCRE regions are independently called, so their 'chrN:start-end'
    strings never match exactly -- overlap must be computed on intervals. Peaks on
    scaffolds absent from `regions` (e.g. GL456216.1) simply don't overlap.
    """
    import pybedtools

    atac_var_names = pd.Index(atac_var_names)
    peaks_bed = peaks_to_bed_df(atac_var_names)
    regions_bed = peaks_to_bed_df(regions)
    if len(peaks_bed) == 0 or len(regions_bed) == 0:
        return np.zeros(len(atac_var_names), dtype=bool)

    peaks_bt = pybedtools.BedTool.from_dataframe(peaks_bed).sort()
    regions_bt = pybedtools.BedTool.from_dataframe(regions_bed[["chrom", "start", "end"]]).sort()
    hits = peaks_bt.intersect(regions_bt, u=True)          # -u: each peak once if it overlaps any region
    overlapping = {iv.name for iv in hits}
    return np.asarray(atac_var_names.isin(overlapping))


def spatial_neighbor_graph(spatial, n_neighs=6):
    """Symmetric kNN connectivity graph over spatial coordinates (S0 = graph weight sum)."""
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(np.asarray(spatial), n_neighbors=n_neighs, mode="connectivity", include_self=False)
    return A.maximum(A.T)


def top_spatially_variable_mask(adata, hv_mask, W_graph, n_top):
    """Boolean mask (len n_vars) keeping the top-`n_top` HVG features by univariate
    spatial Moran's I.

    Selecting spatially-structured features (rather than by dispersion or a cCRE
    overlap) is what lets a spatial-domain MOFA factor form -- and for sparse ATAC
    it strips the ~96% noise peaks that otherwise pin the ATAC view at the Tau floor.
    It is non-circular w.r.t. any downstream cCRE/cell-type claim: peaks are chosen
    for spatial coherence, not for the annotation being tested.
    """
    hv_mask = np.asarray(hv_mask, dtype=bool)
    idx = np.flatnonzero(hv_mask)
    X = adata[:, hv_mask].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    std = X.std(axis=0)
    keep = np.isfinite(X).all(axis=0) & (std > 0)
    X, idx, std = X[:, keep], idx[keep], std[keep]
    Z = (X - X.mean(axis=0)) / std
    n = Z.shape[0]
    S0 = W_graph.sum()
    lag = W_graph @ Z
    moran = (n / S0) * np.einsum("ij,ij->j", Z, lag) / np.einsum("ij,ij->j", Z, Z)
    order = np.argsort(moran)[::-1][: int(n_top)]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[idx[order]] = True
    return mask


def run_trimodal_mofa(
    trimodal_mudata,
    *,
    mudata_source_path=None,
    modalities=DEFAULT_MOFA_MODALITIES,
    mofa_outfile=None,
    n_factors=20,
    max_atac_features=5000,
    atac_ccre_regions=None,
    spatial_top_features=None,
    spatial_n_neighs=6,
    msi_noise_std=0.0,
    gpu_mode=True,
    seed=42,
    winsorize_percentile=0.5,
    overwrite=False,
):
    """
    Run MOFA+ on a trimodal MuData object (RNA, ATAC, MSI).

    Parameters
    ----------
    trimodal_mudata : mu.MuData
        Pre-loaded trimodal MuData (not modified in-place before MOFA prep).
    mudata_source_path : str or Path, optional
        Original .h5mu path; used to derive target_label and default mofa_outfile.

    Returns (trimodal_mudata, mofa_outfile). Factors are written to
    trimodal_mudata.obsm['X_mofa'] and trimodal_mudata.uns['mofa'].

    If ``mofa_outfile`` already exists, training is skipped unless
    ``overwrite=True``. The saved model is loaded later by
    ``load_trained_mofa_for_downstream``.
    """
    if mudata_source_path is not None:
        mudata_source_path = str(mudata_source_path)
        target_label = Path(mudata_source_path).stem.replace("_rna_atac_msi", "")
        if mofa_outfile is None:
            mofa_outfile = mofa_outfile_from_mudata_path(mudata_source_path)
    else:
        target_label = "trimodal"
        if mofa_outfile is None:
            raise ValueError("mofa_outfile is required when mudata_source_path is not provided.")

    if Path(mofa_outfile).exists() and not overwrite:
        print(f"MOFA {target_label}: found existing model at {mofa_outfile}; skipping training.")
        return trimodal_mudata, mofa_outfile

    trimodal_mudata = trimodal_mudata.copy()

    # Spatial-variability feature selection needs a neighbour graph over spot coords.
    spatial_graph = None
    if spatial_top_features:
        if "spatial" not in trimodal_mudata.obsm:
            raise KeyError(
                f"{target_label}: spatial_top_features set but trimodal_mudata.obsm['spatial'] missing."
            )
        spatial_graph = spatial_neighbor_graph(
            trimodal_mudata.obsm["spatial"], n_neighs=spatial_n_neighs
        )

    # Feed MOFA log-normalized RNA (SCT corrected data) instead of the SCT Pearson
    # residuals stored in .X. The residuals carry a heavy positive tail (max ~17.6)
    # that, under a gaussian likelihood, would let a few outlier spots dominate the
    # factors. Done before HVG/variance masking so the variance filter and MOFA both
    # operate on the same log-normalized matrix. ATAC (non-negative log-norm) and MSI
    # (continuous intensities) are left as-is.
    trimodal_mudata.mod["rna"].X = trimodal_mudata.mod["rna"].layers["SCT_data"]

    mofa_feature_masks = {}
    for modality in modalities:
        if modality not in trimodal_mudata.mod:
            raise KeyError(f"{target_label}: modality {modality!r} not in MuData.")
        if modality in {"rna", "atac"}:
            initial_mask = (
                trimodal_mudata.mod[modality].var["highly_variable"]
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            # Preferred: select the top spatially-variable HVG features (by univariate
            # spatial Moran's I). Gives a dopamine-spatial-domain factor with genuine,
            # non-circular ATAC loadings (see top_spatially_variable_mask docstring).
            if spatial_top_features:
                initial_mask = top_spatially_variable_mask(
                    trimodal_mudata.mod[modality], initial_mask, spatial_graph, spatial_top_features
                )
                print(
                    f"MOFA {target_label}: {modality} kept top {int(initial_mask.sum())} "
                    f"spatially-variable HVG features (Moran's I)."
                )
            # Alternative: restrict ATAC to peaks overlapping supplied cCRE regions (e.g.
            # MXD cCREs). NOTE this makes any 'ATAC loads on those cCREs' claim circular.
            elif modality == "atac" and atac_ccre_regions is not None:
                overlap = atac_peaks_overlapping_regions(
                    trimodal_mudata.mod["atac"].var_names, atac_ccre_regions
                )
                print(
                    f"MOFA {target_label}: {int(overlap.sum())} ATAC peaks overlap "
                    f"cCRE regions; {int((initial_mask & overlap).sum())} also highly_variable."
                )
                initial_mask = initial_mask & overlap
        else:
            initial_mask = np.ones(trimodal_mudata.mod[modality].n_vars, dtype=bool)
        mofa_feature_masks[modality] = finite_variable_feature_mask(
            trimodal_mudata.mod[modality],
            initial_mask,
            modality,
        )

    # Cap ATAC to the top-N peaks by normalized dispersion. With ~20k HVG peaks
    # the signal-to-noise per feature is too low for 20 factors to explain any
    # meaningful ATAC variance (Tau stays at the 2.0 floor). Keeping only the most
    # variable peaks improves conditioning without discarding informative signal.
    if max_atac_features and not spatial_top_features and "atac" in mofa_feature_masks:
        atac = trimodal_mudata.mod["atac"]
        selected = np.flatnonzero(mofa_feature_masks["atac"])
        if len(selected) > max_atac_features:
            disp = atac.var["dispersions_norm"].to_numpy()
            top = selected[np.argsort(disp[selected])[::-1][:max_atac_features]]
            new_mask = np.zeros(atac.n_vars, dtype=bool)
            new_mask[top] = True
            print(
                f"MOFA: capped ATAC from {len(selected)} to {max_atac_features} "
                f"features by normalized dispersion."
            )
            mofa_feature_masks["atac"] = new_mask

    mofa_mudata = mu.MuData(
        {
            modality: trimodal_mudata.mod[modality][:, mofa_feature_masks[modality]].copy()
            for modality in modalities
        },
        obs=trimodal_mudata.obs.copy(),
        uns=trimodal_mudata.uns.copy(),
    )
    print(
        f"MOFA {target_label} selected features: "
        + ", ".join(
            f"{modality}={int(mask.sum())}"
            for modality, mask in mofa_feature_masks.items()
        )
    )

    # Winsorize per-feature outlier tails on the dense continuous MSI view(s).
    # msi_student carries an extreme asymmetric left tail (min ~ -17.8 std) that,
    # even in float64, lets a handful of spots dominate factors -- and under
    # float32 it overflows the variational updates into all-NaN weights. RNA/ATAC
    # are non-negative and bounded (~7-8), so they are left untouched.
    if winsorize_percentile and winsorize_percentile > 0:
        for modality in modalities:
            if not modality.startswith("msi"):
                continue
            X = np.asarray(mofa_mudata.mod[modality].X, dtype=np.float64)
            lo = np.percentile(X, winsorize_percentile, axis=0)
            hi = np.percentile(X, 100.0 - winsorize_percentile, axis=0)
            clipped = int((X < lo).sum() + (X > hi).sum())
            mofa_mudata.mod[modality].X = np.clip(X, lo, hi)
            print(
                f"MOFA {target_label}: winsorized {modality} to "
                f"[{winsorize_percentile}, {100.0 - winsorize_percentile}] "
                f"percentile ({clipped} values clipped)."
            )

    # Inject gaussian noise into the MSI view(s) to model imputation uncertainty.
    # msi_student is a deterministic low-rank projected embedding, so MOFA fits it
    # near-perfectly: its Tau runs away (-> thousands) and its precision-weighted
    # pull (~Tau * n_features) dominates RNA/ATAC by orders of magnitude, starving
    # them of factor influence. Adding noise bounds the MSI residual (Tau <~ 1/var),
    # rebalancing the joint objective so all three views inform the shared factors.
    # Noise std is in units of each feature's std (applied after scale_views centers
    # to comparable scales). Tune so MSI Tau lands within ~10x of RNA/ATAC.
    if msi_noise_std and msi_noise_std > 0:
        rng = np.random.default_rng(seed)
        for modality in modalities:
            if not modality.startswith("msi"):
                continue
            X = np.asarray(mofa_mudata.mod[modality].X, dtype=np.float64)
            feature_std = X.std(axis=0, keepdims=True)
            X = X + rng.normal(0.0, msi_noise_std, size=X.shape) * feature_std
            mofa_mudata.mod[modality].X = X
            print(
                f"MOFA {target_label}: added gaussian noise to {modality} "
                f"(std={msi_noise_std} x per-feature std) to cap Tau runaway."
            )

    # mofapy2/mofax may still hold this path open from a prior run in the same kernel.
    close_h5py_handles_for_path(mofa_outfile)

    with threadpool_limits(limits=1):
        mu.tl.mofa(
            mofa_mudata,
            use_var=None,                       # features already masked upstream; do not re-filter
            likelihoods=["gaussian"] * len(mofa_mudata.mod),  # all views continuous (negatives / non-integer) -> Poisson/Bernoulli invalid
            n_factors=n_factors,                # generous; ARD prunes inactive factors
            scale_views=True,                   # views differ ~20x in per-feature variance -> equalize contributions
            center_groups=True,
            ard_weights=True,                   # per-view-per-factor weight pruning
            ard_factors=True,
            spikeslab_weights=True,             # sparse, interpretable loadings (denoises sparse ATAC; readable for downstream Moran's I)
            spikeslab_factors=False,            # keep factor scores dense -> smooth spatial gradients, not spot-level on/off
            convergence_mode="slow",            # accurate factors for a final analysis run
            n_iterations=1000,
            gpu_mode=gpu_mode,
            gpu_device=2,
            use_float32=False,                  # float64 for numerical stability: float32 overflows the variational updates into all-NaN weights on this larger model / heavy-tailed MSI
            seed=seed,
            outfile=mofa_outfile,
            verbose=True,
        )

    trimodal_mudata.obsm["X_mofa"] = mofa_mudata.obsm["X_mofa"]
    trimodal_mudata.uns["mofa"] = mofa_mudata.uns["mofa"]

    return trimodal_mudata, mofa_outfile

#%% extract cCREs related to MXD (ATAC cluster A12/C12)

catlas_supp_tables_path = Path(os.getenv("DATAPATH")) / "CATlas" / "supplementary_tables"
(cCREs_table_path,) = catlas_supp_tables_path.glob("Supplementary Table 8*.txt")

cCREs_table = pd.read_csv(
    cCREs_table_path,
    sep=r"\t|\|",
    engine="python",
    names=["cCRE_region", "cCRE_id", "CellType"],
    skiprows=1,
)

mxd_cCREs = cCREs_table[cCREs_table['CellType'].eq('MXD')]
mxd_ccre_regions = mxd_cCREs['cCRE_region']   # 'chrN:start-end'; overlapped (not string-matched) with ATAC peaks

d2msn_cCREs = cCREs_table[cCREs_table['CellType'].str.contains('D2MSN')]
d2msn_ccre_regions = d2msn_cCREs['cCRE_region']

#%% Load trimodal targets

spatial_trimodal_mudata = mu.read_h5mu(spatial_target_trimodal_mudata_path)
multiome_trimodal_mudata = mu.read_h5mu(multiome_target_trimodal_mudata_path)

## load multiome target RNA with spatial coordinates
multiome_target_rna_with_spatial = sc.read_h5ad(os.path.join(os.getenv("DATAPATH"), "aligned_data", "target_rna_aligned_with_latents.h5ad"), backed='r')

assert multiome_trimodal_mudata.obs_names.equals(multiome_target_rna_with_spatial.obs_names)
multiome_trimodal_mudata.obsm['spatial'] = multiome_target_rna_with_spatial.obsm['spatial']
multiome_trimodal_mudata.mod['msi_student'].obsm["spatial"] = multiome_trimodal_mudata.obsm["spatial"]
sc.pl.embedding(multiome_trimodal_mudata.mod['msi_student'], basis="spatial", color=['msi:Dopamine', 'REF_arc_gex_graphclust_Cluster'], s=80)

#%% Run MOFA+ on targets

# spatial target
spatial_trimodal_mudata, spatial_mofa_outfile = run_trimodal_mofa(
    spatial_trimodal_mudata,
    mudata_source_path=spatial_target_trimodal_mudata_path,
    modalities=("rna", "atac", "msi_teacher"),
)

# multiome target
# Recipe validated to yield a factor with bivariate Moran's I ~ -0.49 vs dopamine
# (matching the ST NMF ~0.48) whose top ATAC loadings are ~2.6x enriched for MXD
# cCREs -- a *non-circular* ATAC link, since peaks are selected for spatial coherence
# (Moran's I), not for cCRE overlap. Requires multiome_trimodal_mudata.obsm['spatial'].
multiome_trimodal_mudata, multiome_mofa_outfile = run_trimodal_mofa(
    multiome_trimodal_mudata,
    mudata_source_path=multiome_target_trimodal_mudata_path,
    modalities=("rna", "atac", "msi_student"),
    spatial_top_features=2000,   # top-N spatially-variable HVG features per rna/atac view
    n_factors=15,
    msi_noise_std=0.5,             # cap msi_student's Tau runaway; tune via Tau readouts
)

# %% load model for downstream analysis
# mofax crashes if any view's features_metadata group is empty (pd.concat on []).
# Patch: write feature names as a placeholder dataset for views missing metadata.
import h5py, mofax as mfx

def patch_mofa_h5py_features_metadata(mofa_outfile):
    """
    Ensure each MOFA+ view has minimal feature metadata for mofax.

    Some mofapy2 runs omit features_metadata entirely, while others create empty
    per-view groups. mofax can stumble on either form when building metadata.
    """
    close_h5py_handles_for_path(mofa_outfile)

    with h5py.File(mofa_outfile, "a") as f:
        features_metadata = f.require_group("features_metadata")
        for view in f["features"].keys():
            view_metadata = features_metadata.require_group(view)
            if "feature_name" not in view_metadata:
                names = f["features"][view][:].astype(str)
                view_metadata.create_dataset("feature_name", data=names.astype("S"))

def load_trained_mofa_for_downstream(
    spatial_mudata_path=spatial_target_trimodal_mudata_path,
    multiome_mudata_path=multiome_target_trimodal_mudata_path,
    *,
    spatial_trimodal_mudata=None,
    multiome_trimodal_mudata=None,
    multiome_spatial_h5ad_path=None,
    plot_multiome_spatial_qc=False,
):
    """Load saved target MOFA+ models and the MuData objects needed downstream.

    Use this after MOFA+ training has already written the `*_mofa_model.hdf5`
    files. It recreates the variable set used by the downstream analysis without
    rerunning `run_trimodal_mofa()`. Pre-loaded MuData objects can be supplied to
    avoid reading the same input files twice.
    """
    spatial_mudata_path = Path(spatial_mudata_path)
    multiome_mudata_path = Path(multiome_mudata_path)
    if multiome_spatial_h5ad_path is None:
        multiome_spatial_h5ad_path = (
            Path(os.getenv("DATAPATH"))
            / "aligned_data"
            / "target_rna_aligned_with_latents.h5ad"
        )

    if spatial_trimodal_mudata is None:
        spatial_trimodal_mudata = mu.read_h5mu(spatial_mudata_path)
    if multiome_trimodal_mudata is None:
        multiome_trimodal_mudata = mu.read_h5mu(multiome_mudata_path)

    if "spatial" not in multiome_trimodal_mudata.obsm:
        multiome_target_rna_with_spatial = sc.read_h5ad(
            multiome_spatial_h5ad_path, backed="r"
        )
        assert multiome_trimodal_mudata.obs_names.equals(
            multiome_target_rna_with_spatial.obs_names
        )
        multiome_trimodal_mudata.obsm["spatial"] = (
            multiome_target_rna_with_spatial.obsm["spatial"]
        )
    multiome_trimodal_mudata.mod["msi_student"].obsm["spatial"] = multiome_trimodal_mudata.obsm["spatial"]

    if plot_multiome_spatial_qc:
        sc.pl.embedding(
            multiome_trimodal_mudata.mod["msi_student"],
            basis="spatial",
            color=["msi:Dopamine", "REF_arc_gex_graphclust_Cluster"],
            s=80,
        )

    spatial_mofa_outfile = mofa_outfile_from_mudata_path(spatial_mudata_path)
    multiome_mofa_outfile = mofa_outfile_from_mudata_path(multiome_mudata_path)
    for mofa_outfile in (spatial_mofa_outfile, multiome_mofa_outfile):
        if not Path(mofa_outfile).exists():
            raise FileNotFoundError(
                f"Missing trained MOFA+ model: {mofa_outfile}. "
                "Run the MOFA+ training cell first, or pass the correct MuData path."
            )
        patch_mofa_h5py_features_metadata(mofa_outfile)

    spatial_mofa = mfx.mofa_model(spatial_mofa_outfile)
    multiome_mofa = mfx.mofa_model(multiome_mofa_outfile)

    return (
        spatial_trimodal_mudata,
        multiome_trimodal_mudata,
        spatial_mofa,
        multiome_mofa,
        spatial_mofa_outfile,
        multiome_mofa_outfile,
    )


(
    spatial_trimodal_mudata,
    multiome_trimodal_mudata,
    spatial_mofa,
    multiome_mofa,
    spatial_mofa_outfile,
    multiome_mofa_outfile,
) = load_trained_mofa_for_downstream(
    spatial_trimodal_mudata=spatial_trimodal_mudata,
    multiome_trimodal_mudata=multiome_trimodal_mudata,
)

#%% explore dopamine MOFA model
def factor_most_associated_with_group(m, group, cluster_label):
    """Return (factor_name, factor_index_0based, per_factor_means) for the factor
    whose mean value in `group` is largest in magnitude.

    This matches exactly what mfx.plot_factors_matrix(agg='mean', group_label=...)
    shows: value[factor, group] = mean factor score across cells in that group.
    Selection is on |mean| (factor sign is arbitrary in MOFA); the signed mean is
    printed so the loading direction is known.
    """
    Z = m.get_factors(df=True)                                  # samples x FactorN
    groups = m.samples_metadata[cluster_label].astype(str)
    group_means = Z.groupby(groups.values).mean()               # group x FactorN
    group = str(group)
    if group not in group_means.index:
        raise KeyError(
            f"group {group!r} not in {cluster_label!r} values: {list(group_means.index)}"
        )
    means = group_means.loc[group]                              # per-factor mean in group
    factor_name = means.abs().idxmax()
    factor_index = list(Z.columns).index(factor_name)
    print(
        f"Factor most associated with group {group}: {factor_name} "
        f"(mean factor score = {means[factor_name]:.3f})"
    )
    return factor_name, factor_index, means


def factor_dopamine_bivariate_moran(m, mudata, *, msi_view="msi_student",
                                    dopamine_feature="msi:Dopamine", n_neighs=6):
    """Per-factor bivariate Moran's I of MOFA factor scores against the dopamine
    spatial lag -- the same spatial metric used for the ST NMF components.

    This is the principled way to pick the dopamine-associated factor here: it asks
    which factor reproduces dopamine's spatial pattern, matching the NMF criterion,
    rather than which factor merely has a high mean score in a cluster. Returns a
    Series indexed by factor name (sorted by |Moran's I| descending).
    """
    spatial = mudata.obsm["spatial"]
    W_graph = spatial_neighbor_graph(spatial, n_neighs=n_neighs)
    S0 = W_graph.sum()
    dop = mudata.mod[msi_view][:, dopamine_feature].X
    dop = np.asarray(dop.todense() if sp.issparse(dop) else dop, dtype=float).ravel()
    zy = (dop - dop.mean()) / dop.std()
    lag_y = W_graph @ zy
    Z = m.get_factors(df=True)
    n = Z.shape[0]
    out = {}
    for factor in Z.columns:
        zf = Z[factor].to_numpy()
        zf = (zf - zf.mean()) / zf.std()
        out[factor] = (n / S0) * (zf @ lag_y) / (zf @ zf)
    moran = pd.Series(out).reindex(Z.columns)
    ranked = moran.reindex(moran.abs().sort_values(ascending=False).index)
    print(f"Top dopamine-associated factor: {ranked.index[0]} "
          f"(bivariate Moran's I = {ranked.iloc[0]:+.3f})")
    return ranked


def explore_dopamine_mofa_model(m, *, msi_view, mudata=None, dopamine_top_group=None,
                                title_prefix="", cluster_label=None):
    """
    Select the dopamine-associated factor and plot MOFA diagnostics.

    Factor selection priority:
      1. `mudata` given -> factor with max |bivariate Moran's I| vs dopamine (spatial;
         matches the ST NMF criterion). Preferred.
      2. `dopamine_top_group` given -> factor with max |mean score| in that cluster.
      3. otherwise -> factor with max |dopamine loading|.

    Returns max_dopamine_weight_index (0-based), max_dopamine_weight_factor (name),
    and dopamine_weights.
    """

    mfx.plot_factors_matrix(
        m, agg="mean",
        linewidths=0.01, linecolor="#FFFFFF33",
        vmax=10,
        group_label=cluster_label,
    )
    plt.show()

    title = f"{title_prefix}: " if title_prefix else ""
    weights = m.get_weights()

    assert m.get_features().loc[:,'feature'].eq('msi:Dopamine').any()
    assert ~np.isnan(weights).any() # no missing weights

    # Per-factor msi:Dopamine loading (diagnostic barplot below).
    dopamine_weights = m.get_weights(views=[msi_view], df=True).loc["msi:Dopamine"]

    # Select the factor to focus on (see docstring for priority).
    factor_names = list(m.get_factors(df=True).columns)
    if mudata is not None:
        dopamine_biv_moran = factor_dopamine_bivariate_moran(m, mudata, msi_view=msi_view)
        max_dopamine_weight_factor = dopamine_biv_moran.index[0]
        max_dopamine_weight_index = factor_names.index(max_dopamine_weight_factor)
        # bar plot of per-factor spatial association with dopamine
        dopamine_biv_moran.reindex(factor_names).plot(kind="barh")
        plt.title(f"{title_prefix}: " if title_prefix else "" "Factor vs dopamine (bivariate Moran's I)")
        plt.xlabel("Bivariate Moran's I"); plt.ylabel("Factor")
        plt.axvline(0, color="k", linestyle="--"); plt.tight_layout(); plt.show()
    elif dopamine_top_group is not None:
        max_dopamine_weight_factor, max_dopamine_weight_index, _ = \
            factor_most_associated_with_group(m, dopamine_top_group, cluster_label)
    else:
        max_dopamine_weight_factor = dopamine_weights.abs().idxmax()
        max_dopamine_weight_index = factor_names.index(max_dopamine_weight_factor)
        print(f"Factor with strongest |dopamine loading|: {max_dopamine_weight_factor}")

    pd.Series(dopamine_weights).plot(kind="barh")
    plt.title(f"{title}Dopamine weights")
    plt.xlabel("Weight")
    plt.ylabel("Factor")
    plt.axvline(0, color="k", linestyle="--")
    plt.show()

    mfx.plot_weights_correlation(m); plt.show()

    mfx.plot_weights(m, n_features=15, views=["rna"], factors=max_dopamine_weight_factor); plt.show()
    mfx.plot_weights(m, n_features=15, views=["atac"], factors=max_dopamine_weight_factor); plt.show()
    mfx.plot_weights(m, n_features=15, views=[msi_view], factors=max_dopamine_weight_factor); plt.show()

    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=15,
        view=[msi_view], y_repel_coef=0.01, x_rank_offset=-150,
    ); plt.show()
    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=10,
        view=["rna"], y_repel_coef=0.01, x_rank_offset=-150,
    ); plt.show()
    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=10,
        view=["atac"], y_repel_coef=0.01, x_rank_offset=-150,
    ); plt.show()

    mfx.plot_weights_heatmap(
        m, n_features=30,
        factors=[max_dopamine_weight_factor],
        view=msi_view,
        xticklabels_size=6, w_abs=True,
        cmap="viridis", cluster_factors=False,
        figsize=(20, 4),
    ); plt.show()

    mfx.plot_factors_scatter(m, color=cluster_label); plt.show()

    mfx.plot_r2_barplot(
        m, group_label=cluster_label,
        factors=[max_dopamine_weight_factor],
    ); plt.show()
    mfx.plot_r2_barplot(
        m,
        factors=[max_dopamine_weight_factor],
        x="Group", groupby="Factor",
        group_label=cluster_label,
        palette="winter",
    ); plt.show()

    '''
    r2_df = m.get_r2(
        group_label='REF_arc_gex_graphclust_Cluster',
        per_factor=True,
        views='atac'
    )
    r2_top_group = r2_df.query('Group == @dopamine_top_group')
    '''
    return max_dopamine_weight_index, max_dopamine_weight_factor, dopamine_weights


# %%

def find_dopamine_top_hit(trimodal_mudata, groupby_key, *, msi_view="msi_student"):
    msi = trimodal_mudata.mod[msi_view]
    msi_features_split = pd.Series(msi.var_names.str.split(':').str[1])
    is_mz_feature = msi_features_split.str.fullmatch(r'\d+(\.\d+)?').fillna(False)
    is_mz_feature.sum(), (~is_mz_feature).sum()  # m/z peaks vs named metabolites (e.g. Dopamine)
    named_msi_features = msi_features_split[~is_mz_feature]
    named_msi_features = ('msi:' + named_msi_features).tolist()

    # rank_genes_groups reads groupby from the modality AnnData, not mudata.obs
    if groupby_key not in msi.obs.columns:
        msi.obs[groupby_key] = trimodal_mudata.obs[groupby_key]
    msi.obs[groupby_key] = msi.obs[groupby_key].astype('category')

    sc.tl.rank_genes_groups(msi, groupby=groupby_key)
    sc.pl.rank_genes_groups_dotplot(msi, var_names=named_msi_features)

    rank_genes_groups_df = sc.get.rank_genes_groups_df(msi, group=None)
    dopamine_top_hit = (
        rank_genes_groups_df[rank_genes_groups_df['names'].eq('msi:Dopamine')]
        .sort_values('scores', ascending=False)
        .iloc[0]
    )
    print(f"msi:Dopamine top hit: cluster {dopamine_top_hit['group']} "
        f"(score={dopamine_top_hit['scores']:.3f}, "
        f"logFC={dopamine_top_hit['logfoldchanges']:.3f}, "
        f"pval_adj={dopamine_top_hit['pvals_adj']:.3g})")
    return dopamine_top_hit

spatial_dopamine_top_hit = find_dopamine_top_hit(
    spatial_trimodal_mudata, 'ATAC_clusters', msi_view='msi_teacher'
)
multiome_dopamine_top_hit = find_dopamine_top_hit(
    multiome_trimodal_mudata, 'REF_arc_gex_graphclust_Cluster', msi_view='msi_student'
)

spatial_dopamine_factor, spatial_dopamine_factor_n, spatial_dopamine_weights = explore_dopamine_mofa_model(
    spatial_mofa,
    msi_view="msi_teacher",
    title_prefix="spatial target",
    cluster_label="ATAC_clusters",
)
multiome_dopamine_factor, multiome_dopamine_factor_n, multiome_dopamine_weights = explore_dopamine_mofa_model(
    multiome_mofa,
    mudata=multiome_trimodal_mudata,   # -> select factor by bivariate Moran's I vs dopamine
    msi_view="msi_student",
    title_prefix="multiome target",
    cluster_label="REF_arc_gex_graphclust_Cluster",
)

# %% create mofa_X anndata object and plot spatial embedding

def create_mofa_adata(mofa, mudata):
    mofa_X = mofa.get_factors()
    mofa_adata = sc.AnnData(
        X=mofa_X,
        obs=mudata.obs.copy(),
        var=pd.DataFrame(index=["Factor " + str(i+1) for i in range(mofa_X.shape[1])])
    )
    if "spatial" in mudata.obsm:
        mofa_adata.obsm["spatial"] = mudata.obsm["spatial"]
        sc.pl.embedding(mofa_adata, basis="spatial", color=mofa_adata.var_names, ncols=4, s=80)
    else:
        print("No spatial embedding found in mudata.obsm")
    return mofa_adata

spatial_mofa_adata = create_mofa_adata(spatial_mofa, spatial_trimodal_mudata)
multiome_mofa_adata = create_mofa_adata(multiome_mofa, multiome_trimodal_mudata)

#%% ATAC cCRE enrichment of the dopamine-associated factor
# Non-circular: ATAC peaks were selected by spatial Moran's I (spatial_top_features),
# NOT by cCRE overlap -- so enrichment of this factor's top ATAC loadings for MXD
# cCREs is a genuine result, not built in by the feature selection.
def atac_ccre_enrichment(mofa, factor, ccre_regions, *, atac_view="atac", top_ns=(25, 50, 100)):
    """Return (atac_loadings, background_overlap_rate, enrichment_df) for `factor`."""
    w = mofa.get_weights(views=[atac_view], factors=factor, df=True).iloc[:, 0]
    bg_rate = atac_peaks_overlapping_regions(w.index.to_numpy(), ccre_regions).mean()
    rows = []
    for N in top_ns:
        top = w.abs().sort_values(ascending=False).head(N).index.to_numpy()
        rate = atac_peaks_overlapping_regions(top, ccre_regions).mean()
        rows.append({"top_N": N, "overlap_frac": rate,
                     "enrichment": rate / bg_rate if bg_rate else np.nan})
    enr = pd.DataFrame(rows)
    print(f"{factor}: background MXD-cCRE overlap = {bg_rate:.1%}")
    print(enr.to_string(index=False))
    return w, bg_rate, enr


mxd_multiome_atac_weights, mxd_multiome_bg_rate, mxd_multiome_enr = atac_ccre_enrichment(
    multiome_mofa, multiome_dopamine_factor_n, mxd_ccre_regions
)

d2msn_multiome_atac_weights, d2msn_multiome_bg_rate, d2msn_multiome_enr = atac_ccre_enrichment(
    multiome_mofa, multiome_dopamine_factor_n, d2msn_ccre_regions
)

def plot_multiome_atac_enrichment(
    multiome_enr,
    multiome_bg_rate,
    multiome_dopamine_factor_n,
    multiome_atac_weights,
    ccre_regions,
    ccre_label,
    atac_peaks_overlapping_regions_func=atac_peaks_overlapping_regions
):
    """
    Plot ATAC cCRE enrichment bar and top-20 peaks for dopamine-associated MOFA factor.
    - multiome_enr: DataFrame with 'top_N', 'overlap_frac', 'enrichment'
    - multiome_bg_rate: float, background overlap rate
    - multiome_dopamine_factor_n: str, factor ID (e.g., "Factor4")
    - multiome_atac_weights: pd.Series of ATAC loadings
    - ccre_regions: cell-type cCRE region list/array
    - ccre_label: label used in figure titles (e.g., "MXD", "D2MSN")
    - atac_peaks_overlapping_regions_func: function for overlap detection (defaults to local)
    """
    # Plot 1: % of top-N ATAC loadings overlapping cell-type cCREs vs the genome-wide background
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        multiome_enr["top_N"].astype(str),
        multiome_enr["overlap_frac"] * 100,
        color="#4C72B0",
        label="top-N ATAC loadings"
    )
    ax.axhline(
        multiome_bg_rate * 100, color="k", ls="--",
        label=f"background ({multiome_bg_rate:.0%})"
    )
    for _, r in multiome_enr.iterrows():
        ax.text(
            str(int(r["top_N"])),
            r["overlap_frac"] * 100 + 1,
            f"{r['enrichment']:.1f}x",
            ha="center"
        )
    ax.set_xlabel("Top-N ATAC peaks by |loading|")
    ax.set_ylabel(f"% overlapping {ccre_label} cCREs")
    ax.set_title(f"{multiome_dopamine_factor_n}: {ccre_label}-cCRE enrichment of ATAC loadings")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: top-20 ATAC loadings, coloured by whether the peak overlaps a cell-type cCRE
    top20 = multiome_atac_weights.reindex(
        multiome_atac_weights.abs().sort_values(ascending=False).head(20).index
    )
    is_ccre = atac_peaks_overlapping_regions_func(top20.index.to_numpy(), ccre_regions)
    fig, ax = plt.subplots(figsize=(6, 6))
    ypos = np.arange(len(top20))[::-1]
    ax.barh(ypos, top20.values, color=np.where(is_ccre, "#C44E52", "#BBBBBB"))
    ax.set_yticks(ypos)
    ax.set_yticklabels(top20.index, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ATAC loading")
    ax.set_title(f"{multiome_dopamine_factor_n} top ATAC peaks (red = {ccre_label} cCRE)")
    plt.tight_layout()
    plt.show()

plot_multiome_atac_enrichment(
    mxd_multiome_enr,
    mxd_multiome_bg_rate,
    multiome_dopamine_factor_n,
    mxd_multiome_atac_weights,
    mxd_ccre_regions,
    "MXD",
)

plot_multiome_atac_enrichment(
    d2msn_multiome_enr,
    d2msn_multiome_bg_rate,
    multiome_dopamine_factor_n,
    d2msn_multiome_atac_weights,
    d2msn_ccre_regions,
    "D2MSN",
)

#%% find top features for the dopamine / C12 factor
C12_factor = multiome_dopamine_factor_n   # Moran-selected dopamine factor (e.g. "Factor4")
top_C12_features = multiome_mofa.get_top_features(
    factors=C12_factor, n_features=25, views=["rna", "atac", "msi_student"]
)
# Informative checks (feature set now spatial-selected, so these are diagnostics, not asserts).
print(f"msi:Dopamine in top features: {np.isin('msi:Dopamine', top_C12_features).item()}")
triplet = {'Pde10a', 'Rgs9', 'Gng7'}  # Fig. 2b gene triplet
print(f"Fig-2b triplet present: {triplet & set(top_C12_features)} (of {triplet})")

chr17_features = top_C12_features[pd.Series(top_C12_features).str.contains('chr17')]
print("chr17 top features:", list(chr17_features))

# %% plot cCRE enrichment of the top C12 features
# GSEA-style running-enrichment curve: rank all ATAC peaks by their signed loading
# on the dopamine factor and test whether cCRE-overlapping peaks concentrate at the top.
# This is the pre-ranked GSEA (GSEAPreranked) statement, complementary to the top-N
# bar plot above -- it uses the full ranking rather than a handful of thresholds.
import gseapy as gp

def plot_atac_ccre_gsea(
    multiome_atac_weights,
    ccre_regions,
    multiome_dopamine_factor_n,
    ccre_label,
    atac_peaks_overlapping_regions_func=atac_peaks_overlapping_regions,
    permutation_num=1000,
    seed=0,
    output_file=None,
):
    """
    Pre-ranked GSEA of cCRE-overlapping peaks against ATAC loadings for the dopamine factor.
    - multiome_atac_weights: pd.Series of ATAC loadings (index = peak IDs)
    - ccre_regions: cell-type cCRE region list/array
    - ccre_label: label used in figure titles (e.g., "MXD", "D2MSN")
    Returns the gseapy prerank result object.
    """
    # Ranked list: peak_id -> signed loading, descending (gseapy re-sorts internally).
    rnk = (
        multiome_atac_weights
        .sort_values(ascending=False)
        .rename("score")
        .rename_axis("peak")
        .reset_index()
    )
    # "Gene set" = peaks overlapping this cell type's cCREs.
    is_ccre = atac_peaks_overlapping_regions_func(
        multiome_atac_weights.index.to_numpy(), ccre_regions
    )
    ccre_peaks = multiome_atac_weights.index[is_ccre].tolist()
    gene_sets = {f"{ccre_label}_cCRE": ccre_peaks}

    pre = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        min_size=1,
        max_size=len(rnk),      # don't drop the set for being "too large"
        permutation_num=permutation_num,
        seed=seed,
        no_plot=True,
        outdir=None,
    )
    print(pre.res2d[["Term", "ES", "NES", "NOM p-val", "FDR q-val"]].to_string(index=False))

    term = f"{ccre_label}_cCRE"
    axes = gp.gseaplot(
        rank_metric=pre.ranking,
        term=term,
        **pre.results[term],
        figsize=(5, 5),
    )
    # gseaplot hard-codes gene-expression labels; relabel for the ATAC-peak context.
    for ax in axes:
        if ax.get_xlabel() == "Gene Rank":
            ax.set_xlabel("Peak Rank")
        if ax.get_ylabel() == "Ranked metric":
            ax.set_ylabel("Ranked weight")
    fig = axes[0].figure
    if output_file is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(output_file)
        plt.close(fig)
    return pre

# Save GSEA plots to cibb overleaf figures directory
overleaf_figures_dir = "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures"
os.makedirs(overleaf_figures_dir, exist_ok=True)

mxd_multiome_gsea = plot_atac_ccre_gsea(
    mxd_multiome_atac_weights,
    mxd_ccre_regions,
    multiome_dopamine_factor_n,
    "MXD",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_mxd_gsea.pdf"),
)

d2msn_multiome_gsea = plot_atac_ccre_gsea(
    d2msn_multiome_atac_weights,
    d2msn_ccre_regions,
    multiome_dopamine_factor_n,
    "D2MSN",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_d2msn_gsea.pdf"),
)


def gsea_leading_edge_peaks(pre, ccre_label):
    """Leading-edge peaks driving the cCRE enrichment for one cell type.

    gseapy's `lead_genes` is the set of query members (here, cell-type cCRE peaks)
    ranked at or before the running-ES maximum -- i.e. up to the rank where the
    enrichment curve inflects from rising to falling. Because the query set is the
    cell-type-specific cCRE peaks, this leading edge is already the intersection of
    "top-ranked by dopamine loading" and "overlaps this cell type's cCREs", and so
    differs between MXD and D2MSN.
    """
    term = f"{ccre_label}_cCRE"
    lead = pre.results[term]["lead_genes"]
    peaks = lead.split(";") if lead else []
    print(f"{ccre_label}: {len(peaks)} leading-edge cCRE peaks (up to ES-curve inflection)")
    return peaks


def gsea_matched_ccre_peaks(pre, ccre_label):
    """All of this cell type's cCRE peaks present in the ranking (the leading edge's
    parent set). Used as the motif-enrichment background so the test contrasts the
    dopamine-factor-driving cCREs against *other cCREs of the same cell type*, rather
    than against generic accessible peaks -- controlling for the fact that cCREs carry
    more TF motifs than average peaks.
    """
    term = f"{ccre_label}_cCRE"
    matched = pre.results[term]["matched_genes"]
    return matched.split(";") if matched else []

#%% TF motif enrichment hand-off (conda env: scenicplus)
# This env lacks MOODS/pyjaspar, so the actual PWM scan + Fisher's-exact test
# runs out-of-process via run_motif_enrichment.py in the `scenicplus` conda env
# (see that script's docstring for the MOODS+pyjaspar design rationale). Unlike
# the spatialMETA hand-off in 0__sea_ad_celltype_tangram_minimal.py, this is a
# one-shot call on an already-small ranked peak list (not a per-donor loop over
# large arrays), so no staging/manifest/caching infrastructure is needed -- just
# write two BEDs, invoke the worker, read back its output TSV.
#
# Foreground = the GSEA leading-edge cCRE peaks for each cell type (the cCRE-
# overlapping peaks ranked up to the running-ES inflection on the dopamine factor);
# background = the same cell type's remaining cCRE peaks. Using the cell-type-specific
# leading edge -- rather than a shared top-N by |loading| -- makes MXD and D2MSN
# select different foregrounds; contrasting against other cCREs of the same type (not
# generic peaks) controls for cCREs' higher baseline motif density, so any enrichment
# reflects the dopamine factor rather than cCRE status. Non-circular for the same
# reason as the cCRE enrichment: peaks were selected by spatial Moran's I, not motifs.
import json
import subprocess

SCENICPLUS_ENV_PYTHON = "/home/mcb/users/dmannk/.conda/envs/scenicplus/bin/python"
MOTIF_ENRICHMENT_WORKER = Path(__file__).resolve().parent / "run_motif_enrichment.py"
MOTIF_ENRICHMENT_OUT_DIR = PROJECTION_DIR / "motif_enrichment"
MM10_FASTA_PATH = Path.home() / ".local/share/genomes/mm10/mm10.fa"


def run_tf_motif_enrichment(
    atac_weights,
    *,
    label,
    foreground_names=None,
    background_names=None,
    top_n=100,
    genome_name="mm10",
    fasta_path=MM10_FASTA_PATH if MM10_FASTA_PATH.exists() else None,
    out_dir=MOTIF_ENRICHMENT_OUT_DIR,
):
    """Hand off a foreground-vs-background motif enrichment test to `scenicplus`.

    foreground = `foreground_names` if given (e.g. the GSEA leading-edge cCRE peaks,
    which differ by cell type), else the top-N peaks by |loading|. background =
    `background_names` if given (e.g. the same cell type's cCRE peaks, so the test
    contrasts dopamine-driving cCREs against other cCREs of that type), else every
    remaining peak with a non-zero loading. Either universe has the foreground removed
    before testing. Returns the enrichment DataFrame (also written to
    out_dir/{label}_motif_enrichment.tsv).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if foreground_names is None:
        foreground_names = atac_weights.abs().sort_values(ascending=False).head(top_n).index
    # Restrict foreground/background to peaks present in the loading vector, and always
    # remove the foreground from the background so the two sets are disjoint.
    foreground_names = atac_weights.index.intersection(pd.Index(foreground_names))
    if background_names is None:
        background_names = atac_weights.index
    background_names = (
        atac_weights.index.intersection(pd.Index(background_names)).difference(foreground_names)
    )

    foreground_bed = peaks_to_bed_df(foreground_names)
    background_bed = peaks_to_bed_df(background_names)
    foreground_bed_path = out_dir / f"{label}_foreground.bed"
    background_bed_path = out_dir / f"{label}_background.bed"
    foreground_bed.to_csv(foreground_bed_path, sep="\t", header=False, index=False)
    background_bed.to_csv(background_bed_path, sep="\t", header=False, index=False)

    output_tsv = out_dir / f"{label}_motif_enrichment.tsv"
    config = {
        "foreground_bed": str(foreground_bed_path),
        "background_bed": str(background_bed_path),
        "output_tsv": str(output_tsv),
        "genome_name": genome_name,
        "fasta_path": str(fasta_path) if fasta_path is not None else None,
    }
    config_path = out_dir / f"{label}_motif_enrichment_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Handing off motif enrichment for {label!r} to scenicplus env "
          f"({len(foreground_bed)} fg / {len(background_bed)} bg peaks)...")
    result = subprocess.run(
        [SCENICPLUS_ENV_PYTHON, str(MOTIF_ENRICHMENT_WORKER), str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = out_dir / f"{label}_motif_enrichment.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"run_motif_enrichment.py failed for {label!r} (see {log_path})."
        )

    enr = pd.read_csv(output_tsv, sep="\t")
    print(f"{label}: top motif hits\n{enr.head(10).to_string(index=False)}")
    return enr


# Foreground = GSEA leading-edge cCRE peaks (cell-type-specific); background = the rest
# of the same cell type's cCRE peaks. Both differ by cell type, so MXD and D2MSN yield
# distinct motif enrichments, and the cCRE-vs-cCRE contrast controls for generic cCRE
# motif density -- isolating what distinguishes the dopamine-driving cCREs.
mxd_leading_edge_peaks = gsea_leading_edge_peaks(mxd_multiome_gsea, "MXD")
d2msn_leading_edge_peaks = gsea_leading_edge_peaks(d2msn_multiome_gsea, "D2MSN")
mxd_ccre_background_peaks = gsea_matched_ccre_peaks(mxd_multiome_gsea, "MXD")
d2msn_ccre_background_peaks = gsea_matched_ccre_peaks(d2msn_multiome_gsea, "D2MSN")

mxd_motif_enrichment = run_tf_motif_enrichment(
    mxd_multiome_atac_weights, label="mxd",
    foreground_names=mxd_leading_edge_peaks,
    background_names=mxd_ccre_background_peaks,
)
d2msn_motif_enrichment = run_tf_motif_enrichment(
    d2msn_multiome_atac_weights, label="d2msn",
    foreground_names=d2msn_leading_edge_peaks,
    background_names=d2msn_ccre_background_peaks,
)

def plot_top_motifs(enr, label, n_top=15, output_file=None):
    from matplotlib.patches import Patch

    top = enr.sort_values("padj").head(n_top).iloc[::-1]

    # Three significance tiers, encoded by lightness (darkest = strongest) so the
    # ordering survives greyscale/colour-blind viewing:
    #   FDR q < 0.05              -> survives multiple-testing correction
    #   nominal p < 0.05 (FDR ns) -> suggestive only
    #   otherwise                 -> not significant
    TIER_COLORS = {"fdr": "#B2182B", "nominal": "#F4A582", "ns": "#BBBBBB"}

    def tier(row):
        if row["padj"] < 0.05:
            return "fdr"
        if row["pvalue"] < 0.05:
            return "nominal"
        return "ns"

    bar_colors = [TIER_COLORS[tier(r)] for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["motif_name"], -np.log10(top["padj"].clip(lower=1e-300)), color=bar_colors)
    # FDR q = 0.05 cutoff on the (already -log10 FDR) x-axis; bars past it are FDR-significant.
    ax.axvline(-np.log10(0.05), color="k", ls="--", lw=0.8, zorder=0)
    ax.set_xlabel("-log10(FDR q-value)")
    ax.set_title(f"{label}: top enriched TF motifs")
    ax.legend(
        handles=[
            Patch(facecolor=TIER_COLORS["fdr"], label="FDR q < 0.05"),
            Patch(facecolor=TIER_COLORS["nominal"], label="nominal p < 0.05"),
            Patch(facecolor=TIER_COLORS["ns"], label="n.s."),
        ],
        fontsize=8, loc="lower right", frameon=False,
    )
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        fig.savefig(output_file)
        plt.close(fig)

plot_top_motifs(
    mxd_motif_enrichment, "MXD",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_mxd_motif_enrichment.pdf"),
)
plot_top_motifs(
    d2msn_motif_enrichment, "D2MSN",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_d2msn_motif_enrichment.pdf"),
)


# %% RNA-side regulon enrichment (TRRUST) vs ATAC motif hits
# Cross-modality check: the ATAC motif enrichment says certain TFs bind the dopamine-
# driving cCREs. Here we test the complementary RNA statement -- are those TFs' *target
# genes* (TRRUST regulons) enriched among the dopamine factor's top RNA loadings? -- and
# report which TFs are hits on BOTH modalities. Default test is ORA (gp.enrichr
# hypergeometric on the top-N factor genes) -- simple and parallel to the ATAC top-N
# foreground; pass method="gsea" for a pre-ranked GSEA on the full signed ranking
# (consistent with the cCRE GSEA above, no gene cutoff).
#
# NOTE: library choice is a coverage/evidence-type trade-off.
#   - Curated/ChIP libraries (TRRUST_Transcription_Factors_2019, ChEA_2022, TF_Perturbations_Followed_by_Expression, TRANSFAC_and_JASPAR_PWMs)
#   give *direct-binding* regulons, which
#     match the ATAC-motif logic best -- but they DON'T cover the D2MSN nuclear-receptor
#     hits (RXRG/RXRB/NR2F1 largely absent; ChEA_2022 has only RXRA). Good for the MXD
#     EGR/KLF/SP story.
#   - Co-expression libraries ("ARCHS4_TFs_Coexp") DO cover all the D2MSN NRs, but are
#     guilt-by-association rather than direct targets -- a softer cross-check for a
#     motif-based claim. Use for the D2MSN NR concordance, and label it as such.
# The parser and GSEA are library-agnostic, so this is a one-line swap; consider running
# both and reporting the concordance per library.

REGULON_LIBRARY = "TRANSFAC_and_JASPAR_PWMs"


def parse_regulon_library(lib, organism=None):
    """Collapse an Enrichr TF library into {TF_symbol: [uppercase targets]}.

    Works across libraries with different term-naming schemes -- TRRUST
    ('TF mouse') and ChEA_2022 ('TF <pmid> <assay> <tissue> <Organism>') both put
    the TF symbol first and the organism last. TF = first token; organism = last
    token. If `organism` is given, keep only matching terms; otherwise keep all and
    union each TF's targets across every experiment/organism (maximises coverage).
    """
    regulons = {}
    for term, genes in lib.items():
        toks = term.split()
        if not toks:
            continue
        tf = toks[0].upper()
        org = toks[-1].lower() if len(toks) > 1 else None
        if organism is not None and org != organism.lower():
            continue
        regulons.setdefault(tf, set()).update(g.upper() for g in genes)
    return {tf: sorted(g) for tf, g in regulons.items()}


def fit_weight_gmm(w, *, seed=0):
    """Fit a 2-component Gaussian mixture to |weight| for one factor.

    Spike-and-slab training (`model_options/spikeslab_weights=True`) makes each
    factor's loadings bimodal: a dense 'spike' of shrunk-out features near 0 plus a
    'slab' of genuine loadings. MOFA doesn't save the per-weight posterior inclusion
    probability, so we recover it empirically -- fitting two Gaussians to |w| and
    treating the higher-mean component as the significant (slab) set. Genes with a
    slab posterior >= 0.5 are 'significant'.

    Returns a dict with the fitted `gm`, absolute weights `aw`, `slab` component index,
    per-gene slab posterior `resp`, boolean `sig` mask, decision `threshold` (smallest
    significant |w|), and `dbic` (1-component BIC minus 2-component BIC; large positive
    => bimodality strongly favoured).
    """
    from sklearn.mixture import GaussianMixture

    aw = np.abs(np.asarray(w, dtype=float))
    X = aw.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=seed, n_init=3).fit(X)
    gm = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=seed, n_init=5).fit(X)
    slab = int(np.argmax(gm.means_.ravel()))
    resp = gm.predict_proba(X)[:, slab]
    sig = resp >= 0.5
    threshold = float(aw[sig].min()) if sig.any() else np.inf
    return {"gm": gm, "aw": aw, "slab": slab, "resp": resp, "sig": sig,
            "threshold": threshold, "dbic": float(g1.bic(X) - gm.bic(X))}


def gmm_significant_weights(w, *, seed=0):
    """Index of genes in the slab (significant) GMM component of |w|. See fit_weight_gmm."""
    fit = fit_weight_gmm(w, seed=seed)
    return pd.Index(w.index)[fit["sig"]]


def plot_weight_gmm(w, *, seed=0, output_file=None, title=None):
    """Two-panel diagnostic of the 2-component GMM fit on |weight| (see fit_weight_gmm).

    Left: histogram of |w| with the spike/slab component densities and their mixture
    (linear y). Right: same on a log y-axis to show the slab fit through the tail. The
    dotted line marks the |w| decision boundary; genes to its right are 'significant'.
    """
    from scipy.stats import norm

    fit = fit_weight_gmm(w, seed=seed)
    gm, aw, slab, sig, thr = fit["gm"], fit["aw"], fit["slab"], fit["sig"], fit["threshold"]
    means = gm.means_.ravel()
    sds = np.sqrt(gm.covariances_.ravel())
    wts = gm.weights_.ravel()
    xs = np.linspace(0, aw.max(), 600)
    comp = [wts[k] * norm.pdf(xs, means[k], sds[k]) for k in range(2)]
    mix = comp[0] + comp[1]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, logy in zip(axes, [False, True]):
        ax.hist(aw, bins=90, density=True, color="#DDDDDD", edgecolor="none")
        for k in range(2):
            ax.plot(xs, comp[k], lw=2, color="#B2182B" if k == slab else "#4C72B0",
                    label="slab (significant)" if k == slab else "spike (background)")
        ax.plot(xs, mix, "k--", lw=1, label="mixture")
        ax.axvline(thr, color="k", ls=":", lw=1)
        ax.set_xlabel("|MOFA weight|")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 1e2)
            ax.set_title("log-y (slab fit)")
        else:
            ax.set_ylabel("density")
            ax.set_title(f"n_sig={int(sig.sum())}  thr={thr:.3f}  ΔBIC={fit['dbic']:.0f}")
    axes[0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Wrote GMM weight-fit diagnostic -> {output_file}")
    return fit


def rna_regulon_enrichment(
    mofa, factor, *, method="ora", rna_view="rna", organism=None,
    library=REGULON_LIBRARY, top_n=200, foreground="gmm", background=None,
    permutation_num=1000, seed=0,
):
    """Regulon enrichment of the dopamine factor's RNA genes.

    method="ora" (default): gp.enrichr hypergeometric test on the top-`top_n` genes
        by |loading|, with `background` as the universe -- fast, and parallel to the
        ATAC top-N-peaks foreground.
    method="gsea": gp.prerank on the full signed ranking (no gene cutoff), consistent
        with the cCRE GSEA above.

    `foreground` (ORA only): how to pick the discrete gene list.
        "gmm" (default): the significant (slab) genes from a 2-component GMM on |w|
            (see fit_weight_gmm) -- a data-driven, factor-specific set that mirrors the
            spike-and-slab structure instead of an arbitrary cutoff.
        "top_n": the top-`top_n` genes by |loading|.

    `background`: iterable of gene symbols defining the enrichment universe (e.g.
        multiome_trimodal_mudata.mod['rna'].var_names to use the whole detected
        transcriptome). Default None -> the factor's own weighted genes. For ORA this
        sets the hypergeometric N directly; for GSEA the extra background genes are
        folded into the ranking at a neutral score of 0 so the universe still matches.
    `organism=None` unions each TF's targets across all experiments/organisms in
    `library`; pass e.g. "mouse" to restrict. Returns a DataFrame normalised to
    columns [Term, pval, padj, ...native...], sorted by padj; `Term` = bare TF symbol.
    """
    w = mofa.get_weights(views=[rna_view], factors=factor, df=True).iloc[:, 0]
    # Regulon targets are UPPERCASE; MOFA mouse symbols are title-case -> uppercase to
    # match. Collapse any post-uppercasing collisions to the strongest-|loading| gene.
    w.index = w.index.str.upper()
    w = w.reindex(w.abs().sort_values(ascending=False).index)
    w = w[~w.index.duplicated(keep="first")]

    # Background universe (uppercased, unique). Default = the factor's weighted genes.
    if background is None:
        background_genes = w.index
    else:
        background_genes = pd.Index([str(g).upper() for g in background]).unique()

    regulons = parse_regulon_library(gp.get_library(library), organism)
    print(f"{library}: {len(regulons)} TF regulons"
          + (f" ({organism})" if organism else " (all organisms)")
          + f"; {len(w)} weighted / {len(background_genes)} background genes; method={method}")

    if method == "ora":
        # Data-driven (GMM slab) or fixed-N foreground; must be a subset of the universe.
        if foreground == "gmm":
            fg = gmm_significant_weights(w, seed=seed)
            print(f"  GMM foreground: {len(fg)} significant genes (slab component)")
        elif foreground == "top_n":
            fg = w.abs().sort_values(ascending=False).head(top_n).index
        else:
            raise ValueError(f"foreground must be 'gmm' or 'top_n', got {foreground!r}")
        top_genes = fg.intersection(background_genes).tolist()
        enr = gp.enrichr(
            gene_list=top_genes, gene_sets=regulons,
            background=background_genes.tolist(), outdir=None, no_plot=True,
        )
        res = enr.results.rename(columns={"P-value": "pval", "Adjusted P-value": "padj"})
    elif method == "gsea":
        # Fold background-only genes into the ranking at a neutral score so the GSEA
        # universe equals `background_genes` (no-op when background defaults to w).
        extra = background_genes.difference(w.index)
        w_ranked = pd.concat([w, pd.Series(0.0, index=extra)]) if len(extra) else w
        rnk = w_ranked.sort_values(ascending=False).rename("score").rename_axis("gene").reset_index()
        pre = gp.prerank(
            rnk=rnk, gene_sets=regulons, min_size=5, max_size=1500,
            permutation_num=permutation_num, seed=seed, no_plot=True, outdir=None,
        )
        res = pre.res2d.rename(columns={"NOM p-val": "pval", "FDR q-val": "padj"})
    else:
        raise ValueError(f"method must be 'ora' or 'gsea', got {method!r}")

    res = res.copy()
    res["pval"] = res["pval"].astype(float)
    res["padj"] = res["padj"].astype(float)
    return res.sort_values("padj").reset_index(drop=True)


def motif_hits_to_tf_symbols(enr, *, use="padj", alpha=0.05):
    """Uppercase TF symbols from significant JASPAR motif rows; splits dimers on '::'."""
    sig = enr.loc[enr[use] < alpha, "motif_name"]
    return {part.upper() for name in sig for part in str(name).split("::")}


# Default ORA (gp.enrichr); pass method="gsea" for the pre-ranked GSEA variant.
# background = whole detected transcriptome (all RNA var_names); pass background=None to
# fall back to the factor's own weighted genes.
rna_background = multiome_trimodal_mudata.mod["rna"].var_names
# ORA foreground = GMM-significant (slab) genes of the factor's RNA loadings.
regulon_res = rna_regulon_enrichment(multiome_mofa, multiome_dopamine_factor_n, foreground="gmm", background=rna_background)
#regulon_res = rna_regulon_enrichment(multiome_mofa, multiome_dopamine_factor_n, method="gsea", background=None)   # GSEA

# Diagnostic: how well the 2-component GMM separates spike (background) from slab.
_rna_w = multiome_mofa.get_weights(views=["rna"], factors=multiome_dopamine_factor_n, df=True).iloc[:, 0]
plot_weight_gmm(
    _rna_w, title=f"Dopamine factor RNA loadings (factor {multiome_dopamine_factor_n})",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_rna_weight_gmm.pdf"),
)

enriched_regulons_fdr = set(regulon_res.loc[regulon_res["padj"] < 0.05, "Term"])
enriched_regulons_nom = set(regulon_res.loc[regulon_res["pval"] < 0.05, "Term"])
_top_cols = [c for c in ["Term", "NES", "Odds Ratio", "Overlap", "pval", "padj"]
             if c in regulon_res.columns]
print("\nTop RNA regulons:\n",
      regulon_res.head(12)[_top_cols].to_string(index=False))

# Cross-modality concordance: regulon enriched on RNA AND motif enriched on ATAC.
for atac_label, atac_enr in [("MXD", mxd_motif_enrichment), ("D2MSN", d2msn_motif_enrichment)]:
    atac_tfs_fdr = motif_hits_to_tf_symbols(atac_enr, use="padj", alpha=0.05)
    atac_tfs_nom = motif_hits_to_tf_symbols(atac_enr, use="pvalue", alpha=0.05)
    print(f"\n[{atac_label}] concordant TFs (enriched on RNA regulon AND ATAC motif):")
    print(f"  RNA-FDR  x ATAC-FDR : {sorted(enriched_regulons_fdr & atac_tfs_fdr)}")
    print(f"  RNA-nom  x ATAC-FDR : {sorted(enriched_regulons_nom & atac_tfs_fdr)}")
    print(f"  RNA-FDR  x ATAC-nom : {sorted(enriched_regulons_fdr & atac_tfs_nom)}")
    print(f"  RNA-nom  x ATAC-nom : {sorted(enriched_regulons_nom & atac_tfs_nom)}")


# Cross-modality concordance tiers, in decreasing stringency (== the priority order used
# to assign each TF its single, most-stringent tier). FDR-significant implies nominally
# significant, so the sets are nested; assigning the first match makes them exclusive.
CONCORDANCE_TIERS = [
    ("RNA-FDR x ATAC-FDR", "#B2182B"),   # both FDR  -> strongest (dark red)
    ("RNA-nom x ATAC-FDR", "#F4A582"),   # ATAC FDR, RNA nominal (light red)
    ("RNA-FDR x ATAC-nom", "#92C5DE"),   # RNA FDR, ATAC nominal (light blue)
    ("RNA-nom x ATAC-nom", "#2166AC"),   # both nominal only (dark blue)
]
CONCORDANCE_COLORS = dict(CONCORDANCE_TIERS)
NOT_CONCORDANT_COLOR = "#DDDDDD"


def assign_concordance_tier(term, rna_fdr, rna_nom, atac_fdr, atac_nom):
    """Most-stringent RNAxATAC concordance tier a TF satisfies, or None (not concordant)."""
    if term in rna_fdr and term in atac_fdr:
        return "RNA-FDR x ATAC-FDR"
    if term in rna_nom and term in atac_fdr:
        return "RNA-nom x ATAC-FDR"
    if term in rna_fdr and term in atac_nom:
        return "RNA-FDR x ATAC-nom"
    if term in rna_nom and term in atac_nom:
        return "RNA-nom x ATAC-nom"
    return None


def plot_regulon_concordance_dotplot(regulon_res, atac_enr, label, *, n_top=20, output_file=None):
    """Enrichr-style dotplot of top RNA regulons, colored by RNAxATAC concordance tier.

    x = -log10(RNA regulon FDR); dot size proportional to the regulon's overlap gene
    count; dot color = the most-stringent concordance tier the TF reaches against this
    cell type's ATAC motif hits (grey if not concordant at all). See CONCORDANCE_TIERS.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rna_fdr = set(regulon_res.loc[regulon_res["padj"] < 0.05, "Term"])
    rna_nom = set(regulon_res.loc[regulon_res["pval"] < 0.05, "Term"])
    atac_fdr = motif_hits_to_tf_symbols(atac_enr, use="padj", alpha=0.05)
    atac_nom = motif_hits_to_tf_symbols(atac_enr, use="pvalue", alpha=0.05)

    top = regulon_res.sort_values("padj").head(n_top).copy()
    top["tier"] = [assign_concordance_tier(t, rna_fdr, rna_nom, atac_fdr, atac_nom)
                   for t in top["Term"]]
    # Overlap "k/n" -> k overlapping genes for the dot size (constant if absent, e.g. GSEA).
    if "Overlap" in top.columns:
        n_genes = top["Overlap"].astype(str).str.split("/").str[0].astype(float).to_numpy()
    else:
        n_genes = np.ones(len(top))
    top = top.iloc[::-1]                      # most significant at the top of the axis
    n_genes = n_genes[::-1]

    smin, smax = 40.0, 400.0
    span = n_genes.max() - n_genes.min()
    sizes = (smin + (n_genes - n_genes.min()) / span * (smax - smin)
             if span > 0 else np.full(len(n_genes), 0.5 * (smin + smax)))
    colors = [CONCORDANCE_COLORS.get(t, NOT_CONCORDANT_COLOR) for t in top["tier"]]
    x = -np.log10(top["padj"].clip(lower=1e-300))
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(6.5, 0.34 * len(top) + 1.6))
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="k", linewidth=0.4, zorder=3)
    ax.axvline(-np.log10(0.05), color="k", ls="--", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(top["Term"])
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlabel("-log10(RNA regulon FDR)")
    ax.set_title(f"RNA regulon enrichment x {label} ATAC motif concordance")

    tier_handles = [Patch(facecolor=c, edgecolor="k", lw=0.4, label=t)
                    for t, c in CONCORDANCE_TIERS]
    tier_handles.append(Patch(facecolor=NOT_CONCORDANT_COLOR, edgecolor="k", lw=0.4,
                              label="not concordant"))
    leg1 = ax.legend(handles=tier_handles, fontsize=7, loc="lower right",
                     frameon=False, title="concordance")
    ax.add_artist(leg1)

    # Size legend: representative overlap gene counts mapped through the same transform.
    refs = np.unique(np.quantile(n_genes, [0.0, 0.5, 1.0]).round().astype(int))
    ref_s = (smin + (refs - n_genes.min()) / span * (smax - smin)
             if span > 0 else np.full(len(refs), 0.5 * (smin + smax)))
    size_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                           markeredgecolor="k", markersize=np.sqrt(s),
                           label=f"{int(v)} genes")
                    for v, s in zip(refs, ref_s)]
    ax.legend(handles=size_handles, fontsize=7, loc="upper left",
              frameon=False, title="overlap", labelspacing=1.2, borderpad=1.0)

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        fig.savefig(output_file)
        plt.close(fig)


# Enrichr-style concordance dotplots: top regulons colored by RNAxATAC concordance tier.
plot_regulon_concordance_dotplot(
    regulon_res, mxd_motif_enrichment, "MXD",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_mxd_regulon_concordance_dotplot.pdf"),
)
plot_regulon_concordance_dotplot(
    regulon_res, d2msn_motif_enrichment, "D2MSN",
    output_file=os.path.join(overleaf_figures_dir, "dopamine_d2msn_regulon_concordance_dotplot.pdf"),
)

# %% Load CATLAS cCRE->gene connections (co-accessibility) as a reusable mapping
# CATLAS whole-mouse-brain co-accessibility links (mm10, matching our peaks): each row
# connects a cCRE (BEDPE anchor B, cols 4-6) to a target gene's promoter (anchor A; gene
# symbol in the col-7 label "cCREs<id>|<Gene>"). Cell-type-specific (e.g. D2MSN1). Loaded
# here as a reusable cCRE->gene table and overlapped onto our ATAC peaks so a peak can be
# resolved to its putative cell-type target gene(s). Downstream use (e.g. chaining
# TF-motif -> cCRE -> gene into empirical regulons) is deferred to a later step.
CATLAS_CONNS_DIR = Path(os.getenv("OUTPATH")) / "catlas_conns"
CATLAS_CONNS_URL = (
    "https://decoder-genetics.wustl.edu/catlasv1/catlas_downloads/"
    "mousebrain/conns/{celltype}.bedpe"
)


def load_catlas_ccre_gene_conns(celltype="D2MSN1", conns_dir=CATLAS_CONNS_DIR):
    """cCRE->gene connections for one CATLAS cell type (downloaded once and cached).

    Returns a DataFrame [chrom, start, end, gene, ccre_id, sign] where chrom/start/end
    are the *cCRE* anchor (BEDPE anchor B, which is the stable locus per cCRE id).
    """
    conns_dir.mkdir(parents=True, exist_ok=True)
    path = conns_dir / f"{celltype}.bedpe"
    if not path.exists():
        import urllib.request
        url = CATLAS_CONNS_URL.format(celltype=celltype)
        print(f"Downloading CATLAS conns for {celltype} -> {path}")
        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["cA", "sA", "eA", "cB", "sB", "eB", "label", "sign"],
    )
    df["ccre_id"] = df["label"].str.split("|").str[0]
    df["gene"] = df["label"].str.split("|").str[1]
    conns = df[["cB", "sB", "eB", "gene", "ccre_id", "sign"]].rename(
        columns={"cB": "chrom", "sB": "start", "eB": "end"}
    )
    print(f"{celltype}: {len(conns)} cCRE->gene links "
          f"({conns['ccre_id'].nunique()} cCREs, {conns['gene'].nunique()} genes)")
    return conns


def peaks_to_linked_genes(atac_var_names, conns):
    """Map ATAC peaks to CATLAS target genes by overlapping peaks with cCRE anchors.

    Returns a tidy DataFrame [peak, gene, ccre_id, sign], one row per peak-gene link.
    """
    import pybedtools

    peaks_bed = peaks_to_bed_df(atac_var_names)
    if len(peaks_bed) == 0 or len(conns) == 0:
        return pd.DataFrame(columns=["peak", "gene", "ccre_id", "sign"])
    peaks_bt = pybedtools.BedTool.from_dataframe(peaks_bed).sort()
    conns_bt = pybedtools.BedTool.from_dataframe(
        conns[["chrom", "start", "end", "gene", "ccre_id", "sign"]]
    ).sort()
    # -wa -wb: emit peak fields (0-3: chrom,start,end,peak) then conn fields
    # (4-9: chrom,start,end,gene,ccre_id,sign) for every overlapping pair.
    hits = peaks_bt.intersect(conns_bt, wa=True, wb=True)
    rows = [
        {"peak": f[3], "gene": f[7], "ccre_id": f[8], "sign": f[9]}
        for f in (iv.fields for iv in hits)
    ]
    return pd.DataFrame(rows, columns=["peak", "gene", "ccre_id", "sign"]).drop_duplicates()


d2msn_conns = load_catlas_ccre_gene_conns("D2MSN1")
# Resolve the dopamine-factor leading-edge cCRE peaks to their D2MSN target genes.
d2msn_leading_edge_gene_links = peaks_to_linked_genes(d2msn_leading_edge_peaks, d2msn_conns)
print(
    f"D2MSN leading edge: {d2msn_leading_edge_gene_links['peak'].nunique()}"
    f"/{len(d2msn_leading_edge_peaks)} peaks linked to "
    f"{d2msn_leading_edge_gene_links['gene'].nunique()} genes"
)


# %% figure functions
# Manuscript figures for the SMA dopamine story. These were first produced by
# standalone scripts that re-loaded everything from disk; here they are rebuilt on
# the objects already in scope, so they always reflect the current run (e.g. the
# no-panel joint_adata / SCT_counts NMF input and the retrained target MOFA models)
# rather than the inputs that happened to be on disk when the figures were first made.
#
# Convention matches plot_atac_ccre_gsea / plot_top_motifs above: pass output_file to
# save, leave it None to show. Each returns the Figure.

FIG_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 12,
    "pdf.fonttype": 42,   # embed TrueType rather than Type-3, for journal submission
    "ps.fonttype": 42,
}


def _fig_finish(fig, output_file):
    if output_file is None:
        plt.show()
    else:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output_file}")
    return fig


def _fig_spatial_scatter(ax, spatial, values, title, *, cbar_label=None,
                         cmap="viridis", s=8, sort_by_value=True):
    """Tissue-map panel: square aspect, inverted y, no frame or ticks."""
    values = np.asarray(values, dtype=float).ravel()
    order = np.argsort(values) if sort_by_value else np.arange(len(values))
    handle = ax.scatter(spatial[order, 0], spatial[order, 1], c=values[order],
                        cmap=cmap, s=s, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    if cbar_label is not None:
        cbar = ax.figure.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    return handle


def _fig_top_loadings(weights, n=12):
    """Top-n features by |loading|, returned ascending so barh reads top-down."""
    return weights.reindex(weights.abs().sort_values(ascending=False).head(n).index).sort_values()


def _fig_msi_bar_colors(features):
    """Red for dopamine and its metabolite 3-MT; green for every other MSI feature."""
    return ["#c0392b" if ("Dopamine" in f or f == "msi:3-MT") else "#7fbf7b" for f in features]


def _fig_sorted_clusters(index, *, strip_prefix=False):
    """Numeric cluster ordering; handles both '3' and 'C12' style labels."""
    def sort_key(label):
        text = str(label)[1:] if strip_prefix else str(label)
        try:
            return (0, int(text))
        except ValueError:
            return (1, str(label))
    return sorted(index, key=sort_key)


def _fig_bare_gene_index(weights, side):
    """Reduce a loading Series to bare gene symbols.

    The two sides disagree on naming: joint_adata stores ST features as 'rna:Pde10a'
    while these target MOFA models store bare 'Pde10a' (other models in this project
    keep the prefix, so neither convention can be assumed). Splitting on the *last*
    ':' normalises both.

    Guards the one case that cannot be normalised: re-running the NMF cell on an
    already-stripped index maps every name to NaN ("Pde10a".split(":")[1]), which
    would otherwise silently empty the intersection.
    """
    weights = pd.Series(weights).copy()
    index = pd.Index(weights.index)
    if index.isna().all():
        raise ValueError(
            f"{side} loadings have an all-null index -- this index was almost certainly "
            "stripped twice ('Pde10a'.split(':')[1] -> NaN). Rebuild it by re-running "
            "the NMF cell against joint_adata, whose ST names carry the 'rna:' prefix."
        )
    weights = weights[index.notna()]
    weights.index = pd.Index([str(gene).split(":")[-1] for gene in weights.index])
    return weights[~weights.index.duplicated()]


def _fig_dopamine_factor(mofa, msi_view, factor=None):
    """Resolve the dopamine factor plus the sign that orients it toward dopamine.

    MOFA factor signs are arbitrary, so every panel gets multiplied by `sign` to keep
    the dopamine loading positive. This makes the teacher and student figures directly
    comparable; when the loading is already positive it is a no-op.
    """
    dopamine_weights = mofa.get_weights(views=[msi_view], df=True).loc["msi:Dopamine"]
    if factor is None:
        factor = dopamine_weights.abs().idxmax()
    sign = float(np.sign(dopamine_weights[factor])) or 1.0
    return factor, sign, dopamine_weights


def _fig_factor_r2(mofa, factor):
    """Per-view R2 (% variance explained) for one factor, averaged over groups."""
    r2 = mofa.get_r2()
    return r2[r2["Factor"] == factor].groupby("View")["R2"].mean()


def _fig_factor_scores_on_spatial(mofa, mudata, factor, sign):
    """Oriented factor scores aligned to the MuData's spatial coordinates."""
    Z = mofa.get_factors(df=True)
    positions = pd.Index(mudata.obs_names.astype(str)).get_indexer(
        Z.index.to_numpy().astype(str)
    )
    if (positions < 0).any():
        raise KeyError("MOFA sample names are not all present in the MuData obs_names.")
    scores = Z[factor].to_numpy() * sign
    return scores, np.asarray(mudata.obsm["spatial"], dtype=float)[positions], positions


def FIG_nmf_dopamine(
    override_nmf_cmp=None, output_file=None
): # can override with '16' for full bi-hemispheric striatal NMF
    nmf_cmp = override_nmf_cmp if override_nmf_cmp is not None else bivariate_moran_I_df.iloc[bivariate_moran_I_df['bivariate_moran_I'].argmax()].name
    nmf_cmp_morans_i = bivariate_moran_I_df.loc[nmf_cmp, 'bivariate_moran_I']

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    sc.pl.embedding(joint_adata, basis='spatial', color='msi:Dopamine', size=100, ax=ax[0], show=False)
    sc.pl.embedding(nmf_adata, basis='spatial', color=nmf_cmp, size=100, ax=ax[1], show=False)
    ax[1].set_title(f'NMF {nmf_cmp} (biv. I = {nmf_cmp_morans_i:.3f})')
    best_nmf_component[~best_nmf_component.index.str.contains('mt-')].sort_values(ascending=True).tail(10).plot(kind='barh', ax=ax[2])
    ax[2].set_title(f'Gene loading for NMF {nmf_cmp}')
    fig.tight_layout(); _fig_finish(fig, output_file)
    return fig, ax



def FIG_mofa_dopamine_factor(
    mofa=None, *, msi_view="msi_student", factor=None, n_top=12, output_file=None,
):
    """The trimodal MOFA+ factor carrying imputed dopamine in the multiome target.

    (a) dopamine's loading on every factor, (b) and (c) that factor's top RNA and MSI
    loadings. Defaults to the Moran-selected factor used by the rest of this script.
    """
    if mofa is None:
        mofa = multiome_mofa
        if factor is None:
            factor = multiome_dopamine_factor_n
    factor, sign, dopamine_weights = _fig_dopamine_factor(mofa, msi_view, factor)

    dopamine_oriented = dopamine_weights * sign
    top_rna = _fig_top_loadings(mofa.get_weights(views=["rna"], df=True)[factor] * sign, n_top)
    top_msi = _fig_top_loadings(mofa.get_weights(views=[msi_view], df=True)[factor] * sign, n_top)

    with plt.rc_context(FIG_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)

        ax = axes[0]
        colors = ["#c0392b" if f == factor else "#95a5a6" for f in dopamine_oriented.index]
        ax.barh(range(len(dopamine_oriented)), dopamine_oriented.values, color=colors)
        ax.set_yticks(range(len(dopamine_oriented)))
        ax.set_yticklabels([f.replace("Factor", "F") for f in dopamine_oriented.index], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("msi:Dopamine loading")
        ax.set_title(f"(a) Dopamine loads on {factor}")

        ax = axes[1]
        ax.barh(range(len(top_rna)), top_rna.values, color="#2c7fb8")
        ax.set_yticks(range(len(top_rna)))
        ax.set_yticklabels(top_rna.index, fontsize=9, fontstyle="italic")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Gene loading")
        ax.set_title(f"(b) Top RNA loadings ({factor})")

        ax = axes[2]
        ax.barh(range(len(top_msi)), top_msi.values, color=_fig_msi_bar_colors(top_msi.index))
        ax.set_yticks(range(len(top_msi)))
        ax.set_yticklabels([f.replace("msi:", "") for f in top_msi.index], fontsize=9)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Metabolite / m/z loading")
        ax.set_title(f"(c) Top MSI loadings ({factor})")

    return _fig_finish(fig, output_file)


def FIG_teacher_mofa_dopamine(
    mofa=None,
    mudata=None,
    *,
    msi_view="msi_teacher",
    factor=None,
    cluster_key="ATAC_clusters",
    highlight_cluster="C12",
    section_label="p22",
    n_top=12,
    output_file=None,
):
    """The teacher-side MOFA+ factor on the spatial target, the counterpart to
    FIG_mofa_dopamine_factor.

    (a) the factor in tissue space, (b) dopamine's loading across factors, (c) mean
    factor score per ATAC cluster -- the MXD cluster is the point of the panel --
    (d)/(e) top RNA and MSI loadings, (f) a numeric summary including per-view R2.
    """
    if mofa is None:
        mofa = spatial_mofa
        if factor is None:
            factor = spatial_dopamine_factor_n
    if mudata is None:
        mudata = spatial_trimodal_mudata
    factor, sign, dopamine_weights = _fig_dopamine_factor(mofa, msi_view, factor)

    scores, spatial, positions = _fig_factor_scores_on_spatial(mofa, mudata, factor, sign)
    clusters = mudata.obs[cluster_key].astype(str).to_numpy()[positions]
    cluster_means = pd.DataFrame({"cluster": clusters, "score": scores}).groupby("cluster")["score"].mean()
    cluster_means = cluster_means.reindex(_fig_sorted_clusters(cluster_means.index, strip_prefix=True))
    top_cluster = cluster_means.idxmax()

    dopamine_oriented = dopamine_weights * sign
    top_rna = _fig_top_loadings(mofa.get_weights(views=["rna"], df=True)[factor] * sign, n_top)
    top_msi = _fig_top_loadings(mofa.get_weights(views=[msi_view], df=True)[factor] * sign, n_top)
    r2 = _fig_factor_r2(mofa, factor)
    runner_up = sorted(cluster_means.values)[-2] if cluster_means.size > 1 else np.nan

    with plt.rc_context(FIG_RCPARAMS):
        fig = plt.figure(figsize=(13, 7.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        ax = fig.add_subplot(gs[0, 0])
        _fig_spatial_scatter(ax, spatial, scores,
                             f"(a) Teacher {factor} on {section_label} spatial",
                             cbar_label="factor score")

        ax = fig.add_subplot(gs[0, 1])
        colors = ["#c0392b" if f == factor else "#95a5a6" for f in dopamine_oriented.index]
        ax.barh(range(len(dopamine_oriented)), dopamine_oriented.values, color=colors)
        ax.set_yticks(range(len(dopamine_oriented)))
        ax.set_yticklabels([f.replace("Factor", "F") for f in dopamine_oriented.index], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("msi:Dopamine loading")
        ax.set_title(f"(b) Dopamine loads on {factor}")

        ax = fig.add_subplot(gs[0, 2])
        colors = [
            "#c0392b" if c == top_cluster else ("#e67e22" if c == highlight_cluster else "#95a5a6")
            for c in cluster_means.index
        ]
        ax.bar(range(len(cluster_means)), cluster_means.values, color=colors)
        ax.set_xticks(range(len(cluster_means)))
        ax.set_xticklabels(cluster_means.index, fontsize=7, rotation=90)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"mean {factor} score")
        ax.set_title(f"(c) Factor by ATAC cluster ({highlight_cluster}=MXD)")

        ax = fig.add_subplot(gs[1, 0])
        ax.barh(range(len(top_rna)), top_rna.values, color="#2c7fb8")
        ax.set_yticks(range(len(top_rna)))
        ax.set_yticklabels(top_rna.index, fontsize=9, fontstyle="italic")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("gene loading")
        ax.set_title(f"(d) Top RNA loadings ({factor})")

        ax = fig.add_subplot(gs[1, 1])
        ax.barh(range(len(top_msi)), top_msi.values, color=_fig_msi_bar_colors(top_msi.index))
        ax.set_yticks(range(len(top_msi)))
        ax.set_yticklabels([f.replace("msi:", "") for f in top_msi.index], fontsize=9)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("metabolite / m/z loading")
        ax.set_title(f"(e) Top MSI loadings ({factor})")

        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        ax.text(0.0, 0.97, f"Teacher {factor} ({section_label} target)",
                fontweight="bold", fontsize=11, va="top")
        summary = (
            f"dopamine loading: {dopamine_oriented[factor]:+.2f}\n"
            f"top ATAC cluster: {top_cluster} (MXD)\n"
            f"  mean {cluster_means[top_cluster]:+.1f} vs {runner_up:+.1f} next\n\n"
            f"R2 (% variance explained):\n"
            f"  MSI  {r2.get(msi_view, np.nan):.1f}\n"
            f"  RNA  {r2.get('rna', np.nan):.2f}\n"
            f"  ATAC {r2.get('atac', np.nan):.3f}"
        )
        ax.text(0.0, 0.82, summary, fontsize=9.5, va="top", family="monospace")

    return _fig_finish(fig, output_file)


def FIG_multiome_dopamine_transfer(
    mudata=None,
    *,
    msi_view="msi_student",
    cluster_key="REF_arc_gex_graphclust_Cluster",
    output_file=None,
):
    """Student-imputed dopamine in the dissociated multiome target lands on one cluster.

    (a) imputed dopamine over the ingest-derived coordinates, (b) the graph-cluster
    occupying that same territory, (c) mean imputed dopamine per cluster -- the
    quantification that makes the co-localisation in (a)/(b) a claim rather than an
    impression.
    """
    from matplotlib.lines import Line2D

    if mudata is None:
        mudata = multiome_trimodal_mudata

    msi = mudata.mod[msi_view]
    dopamine = as_dense(msi[:, "msi:Dopamine"].X).ravel()
    spatial = np.asarray(mudata.obsm["spatial"], dtype=float)
    clusters = (
        msi.obs[cluster_key] if cluster_key in msi.obs.columns else mudata.obs[cluster_key]
    ).astype(str).to_numpy()

    cluster_means = pd.DataFrame({"cluster": clusters, "dopamine": dopamine}).groupby("cluster")["dopamine"].mean()
    cluster_means = cluster_means.reindex(_fig_sorted_clusters(cluster_means.index))
    top_cluster = cluster_means.idxmax()
    is_top = clusters == top_cluster
    print(f"Dopamine-high cluster: {top_cluster} (mean {cluster_means.max():+.3f}; "
          f"next {sorted(cluster_means.values)[-2]:+.3f})")

    with plt.rc_context(FIG_RCPARAMS):
        fig = plt.figure(figsize=(14, 4.4), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])

        ax = fig.add_subplot(gs[0, 0])
        _fig_spatial_scatter(ax, spatial, dopamine,
                             "(a) Student-imputed msi:Dopamine\n(ingest coordinates)",
                             cbar_label="imputed intensity", s=6)

        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(spatial[~is_top, 0], spatial[~is_top, 1], c="#d9d9d9", s=6, linewidths=0)
        ax.scatter(spatial[is_top, 0], spatial[is_top, 1], c="#c0392b", s=8, linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"(b) REF graph-cluster {top_cluster}\n(striatal domain)")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
                       markersize=7, label=f"cluster {top_cluster}"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d9d9d9",
                       markersize=7, label="other"),
            ],
            loc="lower right", fontsize=8, frameon=False,
        )

        ax = fig.add_subplot(gs[0, 2])
        colors = ["#c0392b" if c == top_cluster else "#95a5a6" for c in cluster_means.index]
        ax.bar(range(len(cluster_means)), cluster_means.values, color=colors)
        ax.set_xticks(range(len(cluster_means)))
        ax.set_xticklabels(cluster_means.index, fontsize=7)
        ax.set_xlabel("REF graph-cluster")
        ax.set_ylabel("mean imputed msi:Dopamine")
        ax.set_title(f"(c) Dopamine concentrates in cluster {top_cluster}")

    return _fig_finish(fig, output_file)


def FIG_nmf_mofa_enrichment(
    *,
    nmf_loadings=None,
    models=None,
    top_k=50,
    output_file=None,
):
    """Do the MOFA factors rediscover the ST NMF dopamine program?

    Ranks the NMF program's top-`top_k` genes against each model's factor RNA
    loadings. This is the cross-dataset check: the NMF program is fit on the spatial
    source section, while the factors come from the spatial and multiome *targets*,
    so agreement cannot be an artefact of shared fitting.
    """
    from matplotlib.lines import Line2D
    from sklearn.metrics import roc_auc_score
    from scipy.stats import hypergeom, spearmanr

    if nmf_loadings is None:
        nmf_loadings = best_nmf_component
    if models is None:
        models = [
            ("Teacher", spatial_mofa, "msi_teacher", spatial_dopamine_factor_n),
            ("Student", multiome_mofa, "msi_student", multiome_dopamine_factor_n),
        ]

    nmf_bare = _fig_bare_gene_index(nmf_loadings, "NMF")

    results = []
    for name, mofa, msi_view, factor in models:
        factor, sign, _ = _fig_dopamine_factor(mofa, msi_view, factor)
        rna = _fig_bare_gene_index(
            mofa.get_weights(views=["rna"], df=True)[factor] * sign, f"{name} MOFA RNA"
        )
        shared = nmf_bare.index.intersection(rna.index)
        if len(shared) == 0:
            raise ValueError(
                f"No genes shared between the NMF program and {name} {factor} RNA loadings.\n"
                f"  NMF  ({len(nmf_bare)}): {list(nmf_bare.index[:5])}\n"
                f"  MOFA ({len(rna)}): {list(rna.index[:5])}\n"
                "Both are already stripped of any 'view:' prefix, so this is a genuine "
                "naming mismatch (e.g. case, or mouse vs human symbols), not a prefix issue."
            )
        nmf_shared = nmf_bare.reindex(shared)
        rna_shared = rna.reindex(shared)

        nmf_top = set(nmf_shared.sort_values(ascending=False).head(top_k).index)
        labels = np.fromiter((g in nmf_top for g in shared), dtype=int, count=len(shared))
        auroc = roc_auc_score(labels, rna_shared.values)
        rho, rho_p = spearmanr(nmf_shared.values, rna_shared.values)
        mofa_top = set(rna_shared.sort_values(ascending=False).head(top_k).index)
        overlap = len(nmf_top & mofa_top)
        # P(overlap >= observed) for two top-k draws from the shared gene pool
        hyper_p = float(hypergeom.sf(overlap - 1, len(shared), top_k, top_k))
        expected = top_k ** 2 / len(shared)

        print(f"{name} {factor}: |shared|={len(shared)} AUROC={auroc:.3f} "
              f"rho={rho:.3f} overlap@{top_k}={overlap} (exp {expected:.1f}) p={hyper_p:.2e}")
        results.append({
            "name": name, "auroc": float(auroc), "spearman": float(rho),
            "spearman_p": float(rho_p), "overlap": overlap, "expected": expected,
            "hyper_p": hyper_p, "n_shared": len(shared),
            "top": rna_shared.reindex(nmf_top).dropna().values,
            "rest": rna_shared.drop(index=[g for g in nmf_top if g in rna_shared.index]).values,
        })

    with plt.rc_context(FIG_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

        ax = axes[0]
        for i, res in enumerate(results):
            box = ax.boxplot([res["rest"], res["top"]], positions=[i * 2 + 1, i * 2 + 1.7],
                             widths=0.5, patch_artist=True, showfliers=False)
            for patch, color in zip(box["boxes"], ["#bdbdbd", "#c0392b"]):
                patch.set_facecolor(color)
            ax.text(i * 2 + 1.35, 0.96,
                    f"AUROC={res['auroc']:.2f}\n$p$={res['hyper_p']:.0e}",
                    ha="center", va="top", fontsize=9,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks([i * 2 + 1.35 for i in range(len(results))])
        ax.set_xticklabels([res["name"] for res in results])
        ax.set_ylabel("MOFA factor RNA loading")
        ax.set_title(f"(a) NMF-dopamine top-{top_k} genes\nrank high on the MOFA factor")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#c0392b",
                       markersize=9, label=f"NMF top-{top_k}"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#bdbdbd",
                       markersize=9, label="other shared genes"),
            ],
            fontsize=8, frameon=False, loc="lower right",
        )

        ax = axes[1]
        x = np.arange(len(results))
        width = 0.38
        observed = [res["overlap"] for res in results]
        expected = [res["expected"] for res in results]
        ax.bar(x - width / 2, observed, width, color="#c0392b", label="observed overlap")
        ax.bar(x + width / 2, expected, width, color="#95a5a6", label="expected (random)")
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            ax.text(i - width / 2, obs, str(obs), ha="center", va="bottom", fontsize=9)
            ax.text(i + width / 2, exp, f"{exp:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([res["name"] for res in results])
        ax.set_ylabel(f"genes in top-{top_k} of both")
        ax.set_title("(b) Top-50 overlap: observed vs expected")
        ax.legend(fontsize=8, frameon=False)

    return _fig_finish(fig, output_file)


FIG_nmf_dopamine(
    output_file=os.path.join(overleaf_figures_dir, "sma_dopamine_nmf_xcorr.pdf"),
)
FIG_mofa_dopamine_factor(
    output_file=os.path.join(overleaf_figures_dir, "sma_dopamine_mofa_factor.pdf"),
)
FIG_teacher_mofa_dopamine(
    output_file=os.path.join(overleaf_figures_dir, "sma_dopamine_teacher_mofa.pdf"),
)
FIG_multiome_dopamine_transfer(
    output_file=os.path.join(overleaf_figures_dir, "sma_dopamine_multiome_transfer.pdf"),
)
FIG_nmf_mofa_enrichment(
    output_file=os.path.join(overleaf_figures_dir, "sma_dopamine_nmf_mofa_enrichment.pdf"),
)

# %%
