#%% Load data
# conda env: eclare_env

import os
for thread_env_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[thread_env_var] = "1"

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env", override=True)

import sys
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
joint_adata = sc.read_h5ad(MODEL_DIR / "joint_adata.h5ad")

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
x_visium = joint_adata[:,st_features].layers['normalized'].toarray()
x_dopamine = joint_adata[:, 'msi:Dopamine'].X.toarray().squeeze()
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
    trimodal_mudata_path,
    *,
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

    Returns (trimodal_mudata, mofa_outfile). Factors are written to
    trimodal_mudata.obsm['X_mofa'] and trimodal_mudata.uns['mofa'].
    """
    trimodal_mudata_path = str(trimodal_mudata_path)
    target_label = Path(trimodal_mudata_path).stem.replace("_rna_atac_msi", "")
    if mofa_outfile is None:
        mofa_outfile = mofa_outfile_from_mudata_path(trimodal_mudata_path)

    trimodal_mudata = mu.read_h5mu(trimodal_mudata_path)

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

#%% Run MOFA+ on targets

trimodal_mudata, spatial_mofa_outfile = run_trimodal_mofa(
    spatial_target_trimodal_mudata_path,
    modalities=("rna", "atac", "msi_teacher"),
    atac_ccre_regions=mxd_ccre_regions,
)
multiome_trimodal_mudata, multiome_mofa_outfile = run_trimodal_mofa(
    multiome_target_trimodal_mudata_path,
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
def explore_dopamine_mofa_model(m, *, msi_view, title_prefix=""):
    """
    Find the factor with strongest dopamine loading and plot MOFA diagnostics.

    Returns max_dopamine_weight_index (0-based), max_dopamine_weight_factor (1-based),
    and dopamine_weights.
    """
    title = f"{title_prefix}: " if title_prefix else ""
    weights = m.get_weights()

    assert m.get_features().loc[:,'feature'].eq('msi:Dopamine').any()
    assert ~np.isnan(weights).any() # no missing weights

    norm_weights = weights / np.max(np.abs(weights), axis=0)
    dopamine_index = m.get_features().loc[:, "feature"].eq("msi:Dopamine").idxmax()
    dopamine_weights = norm_weights[dopamine_index]
    max_dopamine_weight_index = int(dopamine_weights.argmax())
    max_dopamine_weight = float(dopamine_weights.max())
    max_dopamine_weight_factor = max_dopamine_weight_index + 1

    pd.Series(dopamine_weights).plot(kind="barh")
    plt.title(f"{title}Dopamine weights")
    plt.xlabel("Weight")
    plt.ylabel("Factor")
    plt.axvline(0, color="k", linestyle="--")
    plt.show()

    mfx.plot_weights_correlation(m)

    mfx.plot_weights(m, n_features=15, views=["rna"], factors=max_dopamine_weight_index)
    mfx.plot_weights(m, n_features=15, views=["atac"], factors=max_dopamine_weight_index)
    mfx.plot_weights(m, n_features=15, views=[msi_view], factors=max_dopamine_weight_index)

    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=15,
        view=[msi_view], y_repel_coef=0.01, x_rank_offset=-150,
    )
    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=10,
        view=["rna"], y_repel_coef=0.01, x_rank_offset=-150,
    )
    mfx.plot_weights_ranked(
        m, factor=max_dopamine_weight_factor, n_features=10,
        view=["atac"], y_repel_coef=0.01, x_rank_offset=-150,
    )

    mfx.plot_weights_heatmap(
        m, n_features=30,
        factors=[max_dopamine_weight_index],
        view=msi_view,
        xticklabels_size=6, w_abs=True,
        cmap="viridis", cluster_factors=False,
        figsize=(20, 4),
    )

    mfx.plot_factors_scatter(m, color="ATAC_clusters")

    mfx.plot_r2_barplot(
        m, group_label="ATAC_clusters",
        factors=[max_dopamine_weight_index, max_dopamine_weight_factor],
    )
    mfx.plot_r2_barplot(
        m,
        factors=[max_dopamine_weight_index, max_dopamine_weight_factor],
        x="Group", groupby="Factor",
        group_label="ATAC_clusters",
        palette="winter",
    )
    mfx.plot_factors_matrix(
        m, agg="mean",
        linewidths=0.01, linecolor="#FFFFFF33",
        vmax=10,
        group_label="ATAC_clusters",
    )

    return max_dopamine_weight_index, max_dopamine_weight_factor, dopamine_weights


# %%

spatial_dopamine_factor, spatial_dopamine_factor_n, spatial_dopamine_weights = explore_dopamine_mofa_model(
    spatial_mofa,
    msi_view="msi_teacher",
    title_prefix="spatial target",
)
multiome_dopamine_factor, multiome_dopamine_factor_n, multiome_dopamine_weights = explore_dopamine_mofa_model(
    multiome_mofa,
    msi_view="msi_student",
    title_prefix="multiome target",
)

# downstream cells use spatial target MOFA model
#m = spatial_mofa
m = multiome_mofa
max_dopamine_weight_index = spatial_dopamine_factor
max_dopamine_weight_factor = spatial_dopamine_factor_n
dopamine_weights = spatial_dopamine_weights

# %% create mofa_X anndata object

mofa_X = m.get_factors()
mofa_adata = sc.AnnData(
    X=mofa_X,
    obs=trimodal_mudata.obs.copy(),
    var=pd.DataFrame(index=["Factor " + str(i+1) for i in range(mofa_X.shape[1])])
)
mofa_adata.obsm["spatial"] = trimodal_mudata.obsm["spatial"]
sc.pl.embedding(mofa_adata, basis="spatial", color=mofa_adata.var_names, ncols=4, s=80)

C12_factor = "Factor 3" # factor with clear pattern for C12/A12 ATAC cluster
ax = mfx.plot_weights(m, n_features=15, views=["rna"], factors=C12_factor)
ax = mfx.plot_weights(m, n_features=15, views=["atac"], factors=C12_factor)
ax = mfx.plot_weights(m, n_features=15, views=["msi_teacher"], factors=C12_factor)

top_C12_features = m.get_top_features(factors=C12_factor, n_features=25, views=["rna", "atac", "msi_teacher"])
assert np.isin('msi:Dopamine', top_C12_features).item()
assert set(['Pde10a','Rgs9','Gng7']) <= set(top_C12_features) # the same gene triplet used in Fig. 2b (left side)

chr17_features = top_C12_features[pd.Series(top_C12_features).str.contains('chr17')]
print(list(chr17_features))

# %%