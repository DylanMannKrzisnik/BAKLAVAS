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

# %%
import muon as mu
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

trimodal_mudata_path = os.path.join(
    os.getenv("OUTPATH"),
    "spatialjepa_projection",
    "V11L12-109_B1",
    "spatial_target_rna_atac_msi.h5mu",
)
trimodal_mudata = mu.read_h5mu(trimodal_mudata_path)
MOFA_MODALITIES = ("rna", "atac", "msi_teacher")

# Feed MOFA log-normalized RNA (SCT corrected data) instead of the SCT Pearson
# residuals stored in .X. The residuals carry a heavy positive tail (max ~17.6)
# that, under a gaussian likelihood, would let a few outlier spots dominate the
# factors. Done before HVG/variance masking so the variance filter and MOFA both
# operate on the same log-normalized matrix. ATAC (non-negative log-norm) and MSI
# (continuous intensities) are left as-is.
trimodal_mudata.mod["rna"].X = trimodal_mudata.mod["rna"].layers["SCT_data"]


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


mofa_feature_masks = {}
for modality in MOFA_MODALITIES:
    if modality in {"rna", "atac"}:
        initial_mask = (
            trimodal_mudata.mod[modality].var["highly_variable"]
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    else:
        initial_mask = np.ones(trimodal_mudata.mod[modality].n_vars, dtype=bool)
    mofa_feature_masks[modality] = finite_variable_feature_mask(
        trimodal_mudata.mod[modality],
        initial_mask,
        modality,
    )

mofa_mudata = mu.MuData(
    {modality: adata[:, mofa_feature_masks[modality]].copy()
     for modality, adata in trimodal_mudata.mod.items()
     if modality in MOFA_MODALITIES},
    obs=trimodal_mudata.obs.copy(),
    uns=trimodal_mudata.uns.copy(),
)
print(
    "MOFA selected features: "
    + ", ".join(
        f"{modality}={int(mask.sum())}"
        for modality, mask in mofa_feature_masks.items()
    )
)

mofa_outfile = os.path.join(
    os.getenv("OUTPATH"),
    "spatialjepa_projection",
    "V11L12-109_B1",
    "mofa_model.hdf5",
)

with threadpool_limits(limits=1):
    mu.tl.mofa(
        mofa_mudata,
        use_var=None,                       # features already masked upstream; do not re-filter
        likelihoods=["gaussian"] * len(mofa_mudata.mod),  # all views continuous (negatives / non-integer) -> Poisson/Bernoulli invalid
        n_factors=20,                       # generous; ARD prunes inactive factors
        scale_views=True,                   # views differ ~20x in per-feature variance -> equalize contributions
        center_groups=True,
        ard_weights=True,                   # per-view-per-factor weight pruning
        ard_factors=True,
        spikeslab_weights=True,             # sparse, interpretable loadings (denoises sparse ATAC; readable for downstream Moran's I)
        spikeslab_factors=False,            # keep factor scores dense -> smooth spatial gradients, not spot-level on/off
        convergence_mode="slow",            # accurate factors for a final analysis run
        n_iterations=1000,
        gpu_mode=True,
        use_float32=True,                   # pairs with GPU: less memory / faster for the large ATAC view
        seed=42,
        outfile=mofa_outfile,
    )

trimodal_mudata.obsm["X_mofa"] = mofa_mudata.obsm["X_mofa"]
trimodal_mudata.uns["mofa"] = mofa_mudata.uns["mofa"]

# %%
