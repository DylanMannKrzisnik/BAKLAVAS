#%% Load data
# conda env: nichecompass_liana

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")

import os
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

x_visium = joint_adata[:,joint_adata.var['type'].eq('ST')].layers['normalized'].toarray()
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