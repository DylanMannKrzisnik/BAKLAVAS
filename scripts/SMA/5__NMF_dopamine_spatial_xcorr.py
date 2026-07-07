#%% Load data
# conda env: eclare_env

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
x_visium = as_dense(joint_adata[:, st_features].layers['normalized'])
x_dopamine = as_dense(joint_adata[:, 'msi:Dopamine'].X).squeeze()
zy = zscore(x_dopamine)                         # the "lagged" variable
lag_y = W @ zy                                   # spatial lag of dopamine

# Poisson NMF with KL loss
nmf_kwargs = dict(
    init='nndsvda',
    solver='mu',
    beta_loss='kullback-leibler',
    max_iter=1000,   # mu converges slower than cd's default 200
    tol=1e-4,
    random_state=0,
)

sweep_n_components = np.arange(1, 30)
max_bivariate_moran_Is = []
for n_components in tqdm(sweep_n_components):

    nmf_adata, _ = fit_nmf_spatial(x_visium, n_components, joint_adata.obsm["spatial"], **nmf_kwargs)
    bivariate_moran_I = bivariate_morans_i(nmf_adata.X, lag_y, n, S0)
    max_bivariate_moran_Is.append(np.max(bivariate_moran_I))

plt.figure(figsize=(10, 5))
plt.plot(sweep_n_components, max_bivariate_moran_Is, marker='o')
plt.xlabel("Number of NMF components")
plt.ylabel("Max bivariate Moran's I")
plt.title("Max bivariate Moran's I vs. Number of NMF components")

best_n_components = sweep_n_components[np.argmax(max_bivariate_moran_Is)]
print(f"Best number of NMF components: {best_n_components}")

nmf_adata, nmf = fit_nmf_spatial(x_visium, best_n_components, joint_adata.obsm["spatial"], **nmf_kwargs)
H = nmf.components_
bivariate_moran_I = bivariate_morans_i(nmf_adata.X, lag_y, n, S0)

# Plot NMF components as subfigures in the same Scanpy figure.
nmf_components = list(nmf_adata.var_names)
nmf_component_titles = [
    f"{comp} (biv. I = {bivariate_moran_I[k]:.3f})"
    for k, comp in enumerate(nmf_components)
]
best_bivariate_moran_idx = int(np.argmax(bivariate_moran_I))
best_bivariate_moran_component = nmf_components[best_bivariate_moran_idx]
best_bivariate_moran_score = bivariate_moran_I[best_bivariate_moran_idx]
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
bivariate_moran_I_df = pd.DataFrame(bivariate_moran_I, index=nmf_components).sort_values(0, ascending=True).rename(columns={0: 'bivariate_moran_I'})
best_nmf_component = pd.Series(nmf.components_[best_bivariate_moran_idx], index=st_features)
best_nmf_component.index = best_nmf_component.index.str.split(':').str[1]

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


def run_trimodal_mofa(
    trimodal_mudata,
    *,
    mudata_source_path=None,
    modalities=DEFAULT_MOFA_MODALITIES,
    mofa_outfile=None,
    n_factors=20,
    max_atac_features=5000,
    atac_ccre_regions=None,
    msi_noise_std=0.0,
    gpu_mode=True,
    seed=42,
    winsorize_percentile=0.5,
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

    trimodal_mudata = trimodal_mudata.copy()

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
            # Restrict ATAC to peaks overlapping the supplied cCRE regions (e.g. MXD
            # cCREs marking ATAC cluster C12). Biology-driven selection focuses the
            # view on the target signal instead of top global-dispersion peaks.
            if modality == "atac" and atac_ccre_regions is not None:
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
    if max_atac_features and "atac" in mofa_feature_masks:
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

spatial_trimodal_mudata, spatial_mofa_outfile = run_trimodal_mofa(
    spatial_trimodal_mudata,
    mudata_source_path=spatial_target_trimodal_mudata_path,
    modalities=("rna", "atac", "msi_teacher"),
    atac_ccre_regions=mxd_ccre_regions,
)
multiome_trimodal_mudata, multiome_mofa_outfile = run_trimodal_mofa(
    multiome_trimodal_mudata,
    mudata_source_path=multiome_target_trimodal_mudata_path,
    modalities=("rna", "atac", "msi_student"),
    atac_ccre_regions=mxd_ccre_regions,
    max_atac_features=None,
    n_factors=20,
    msi_noise_std=1,   # cap msi_student's Tau runaway; tune via Tau readouts
)

# %% load model for downstream analysis
# mofax crashes if any view's features_metadata group is empty (pd.concat on []).
# Patch: write feature names as a placeholder dataset for views missing metadata.
import gc, h5py, mofax as mfx

# mofapy2 leaves a read-only h5py handle open on the output file after training; close it first
def patch_mofa_h5py_features_metadata(mofa_outfile):
    """
    Ensure each MOFA+ view has minimal feature metadata for mofax.

    Some mofapy2 runs omit features_metadata entirely, while others create empty
    per-view groups. mofax can stumble on either form when building metadata.
    """
    mofa_outfile = str(mofa_outfile)
    for obj in gc.get_objects():
        try:
            if isinstance(obj, h5py.File) and obj.id.valid and obj.filename == mofa_outfile:
                obj.close()
        except Exception:
            pass

    with h5py.File(mofa_outfile, "a") as f:
        features_metadata = f.require_group("features_metadata")
        for view in f["features"].keys():
            view_metadata = features_metadata.require_group(view)
            if "feature_name" not in view_metadata:
                names = f["features"][view][:].astype(str)
                view_metadata.create_dataset("feature_name", data=names.astype("S"))

## load spatial target MOFA+ model
patch_mofa_h5py_features_metadata(spatial_mofa_outfile)
spatial_mofa = mfx.mofa_model(spatial_mofa_outfile)

## load multiome target MOFA+ model
patch_mofa_h5py_features_metadata(multiome_mofa_outfile)
multiome_mofa = mfx.mofa_model(multiome_mofa_outfile)

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


def explore_dopamine_mofa_model(m, *, msi_view, dopamine_top_group=None, title_prefix="", cluster_label=None):
    """
    Find the factor with strongest dopamine loading and plot MOFA diagnostics.

    Returns max_dopamine_weight_index (0-based), max_dopamine_weight_factor (1-based),
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

    # Select the factor to focus on. Primary: the factor most associated with the
    # cluster where dopamine is the top hit (from find_dopamine_top_hit). Fallback
    # (no group supplied): the factor with the strongest |dopamine loading|.
    if dopamine_top_group is not None:
        max_dopamine_weight_factor, max_dopamine_weight_index, _ = \
            factor_most_associated_with_group(m, dopamine_top_group, cluster_label)
    else:
        max_dopamine_weight_factor = dopamine_weights.abs().idxmax()
        max_dopamine_weight_index = list(m.get_factors(df=True).columns).index(max_dopamine_weight_factor)
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

def find_dopamine_top_hit(trimodal_mudata):
    msi_features_split = pd.Series(trimodal_mudata.mod['msi_student'].var_names.str.split(':').str[1])
    is_mz_feature = msi_features_split.str.fullmatch(r'\d+(\.\d+)?').fillna(False)
    is_mz_feature.sum(), (~is_mz_feature).sum()  # m/z peaks vs named metabolites (e.g. Dopamine)
    named_msi_features = msi_features_split[~is_mz_feature]
    named_msi_features = ('msi:' + named_msi_features).tolist()

    sc.tl.rank_genes_groups(multiome_trimodal_mudata.mod['msi_student'], groupby='REF_arc_gex_graphclust_Cluster')
    sc.pl.rank_genes_groups_dotplot(multiome_trimodal_mudata.mod['msi_student'], var_names=named_msi_features)

    rank_genes_groups_df = sc.get.rank_genes_groups_df(multiome_trimodal_mudata.mod['msi_student'], group=None)
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

spatial_dopamine_top_hit = find_dopamine_top_hit(spatial_trimodal_mudata)
multiome_dopamine_top_hit = find_dopamine_top_hit(multiome_trimodal_mudata)

spatial_dopamine_factor, spatial_dopamine_factor_n, spatial_dopamine_weights = explore_dopamine_mofa_model(
    spatial_mofa,
    msi_view="msi_teacher",
    title_prefix="spatial target",
    cluster_label="ATAC_clusters",
)
multiome_dopamine_factor, multiome_dopamine_factor_n, multiome_dopamine_weights = explore_dopamine_mofa_model(
    multiome_mofa,
    dopamine_top_group=str(multiome_dopamine_top_hit['group']),
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
    sc.pl.embedding(mofa_adata, basis="spatial", color=mofa_adata.var_names, ncols=4, s=80, vmax=5)
    return multiome_mofa_adata

spatial_mofa_adata = create_mofa_adata(spatial_mofa, spatial_trimodal_mudata)
multiome_mofa_adata = create_mofa_adata(multiome_mofa, multiome_trimodal_mudata)

#%% find top features for C12/A12 ATAC cluster
C12_factor = "Factor 3" # factor with clear pattern for C12/A12 ATAC cluster
top_C12_features = multiome_mofa.get_top_features(factors=C12_factor, n_features=25, views=["rna", "atac", "msi_student"])
assert np.isin('msi:Dopamine', top_C12_features).item()
assert set(['Pde10a','Rgs9','Gng7']) <= set(top_C12_features) # the same gene triplet used in Fig. 2b (left side)

chr17_features = top_C12_features[pd.Series(top_C12_features).str.contains('chr17')]
print(list(chr17_features))

# %%
