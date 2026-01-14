#%%
import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
import spatialdata as sd
import anndata as ad
from spatialdata.models import Image2DModel, PointsModel
from skimage.io import imread
from scipy.sparse import issparse
import seaborn as sns

import scenvi
from envi_utils import envi_train_with_logger, parse_envi_log, read_envi_predictions
from data_utils import multimodal_latents_adata

datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/human"

#%% load data

#adata = ad.read_h5ad(os.path.join(datapath, "brain_spatial.h5ad"))
spatial = ad.read_h5ad(os.path.join(datapath, "Spatial.h5ad"))
rna = ad.read_h5ad(os.path.join(datapath, "SCT.h5ad"))
atac = ad.read_h5ad(os.path.join(datapath, "ATAC.h5ad"))
peaks = ad.read_h5ad(os.path.join(datapath, "peaks.h5ad"))

## confirm that all RNA genes are also in the spatial data (though not true the other way around)
assert rna.var_names.isin(np.intersect1d(rna.var_names, spatial.var_names)).all().item()
assert (rna.obs_names == peaks.obs_names).all().item()
assert (rna.obs_names == spatial.obs_names).all().item()

spatial_genes_not_in_rna = spatial.var_names[~spatial.var_names.isin(rna.var_names)]
print(f"Number of spatial genes not in RNA: {len(spatial_genes_not_in_rna)}")

## load tissue hires image
img = imread(os.path.join(datapath, "tissue_hires_image.png"))
coords = pd.read_csv(os.path.join(datapath, "spatial_coords.csv"), index_col=0).rename(columns={"imagerow": "x", "imagecol": "y"})
spatial.obsm['spatial'] = coords.values

#%% plot clusters using spatial coordinates
fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Histology')
ax[0].axis('off')

ax[1].imshow(img, cmap='gray')
rna_clusters = spatial.obs['RNA_clusters']
unique_clusters = pd.Series(rna_clusters).unique()
cluster_to_code = {cluster: i for i, cluster in enumerate(sorted(unique_clusters))}
cluster_codes = pd.Series(rna_clusters).map(cluster_to_code).values
scatter = ax[1].scatter(coords["y"], coords["x"], s=50, c=cluster_codes, cmap='Set1', alpha=0.6, vmin=-0.5, vmax=len(unique_clusters)-0.5)
ax[1].set_title('RNA Clusters')
ax[1].axis('off')
#cbar = plt.colorbar(scatter, ticks=range(len(unique_clusters)))
#cbar.set_ticklabels(sorted(unique_clusters))

ax[2].imshow(img, cmap='gray')
atac_clusters = spatial.obs['ATAC_clusters']
unique_clusters = pd.Series(atac_clusters).unique()
cluster_to_code = {cluster: i for i, cluster in enumerate(sorted(unique_clusters))}
cluster_codes = pd.Series(atac_clusters).map(cluster_to_code).values
scatter = ax[2].scatter(coords["y"], coords["x"], s=50, c=cluster_codes, cmap='Set1', alpha=0.6, vmin=-0.5, vmax=len(unique_clusters)-0.5)
ax[2].set_title('ATAC Clusters')
ax[2].axis('off')
#cbar = plt.colorbar(scatter, ticks=range(len(unique_clusters)))
#cbar.set_ticklabels(sorted(unique_clusters))

plt.tight_layout()
plt.show()

#%% initialize ENVI models
rna.X = rna.raw.X

if issparse(rna.X):
    rna.X = rna.X.toarray()
if issparse(atac.X):
    atac.X = atac.X.toarray()
if issparse(spatial.X):
    spatial.X = spatial.X.toarray()

envi_model_rna = scenvi.ENVI(spatial_data = spatial, sc_data = rna, covet_batch_size = 256)
envi_model_atac = scenvi.ENVI(spatial_data = spatial, sc_data = atac, covet_batch_size = 256)


#%% train ENVI models with logger

log_output_path = "/home/mcb/users/dmannk/BAKLAVA_base/outputs/log/Spatial_ATAC_RNA/"
os.makedirs(log_output_path, exist_ok=True)

envi_model_atac = envi_train_with_logger(
    envi_model_atac,
    os.path.join(log_output_path, "envi_atac_loss.txt"),
    training_steps=10000
)
envi_model_atac.impute_genes()
envi_model_atac.infer_niche_covet()
#envi_model_atac.infer_niche_celltype()

envi_model_rna = envi_train_with_logger(
    envi_model_rna,
    os.path.join(log_output_path, "envi_rna_loss.txt"),
    training_steps=10000
)
envi_model_rna.impute_genes()
envi_model_rna.infer_niche_covet()
#envi_model_rna.infer_niche_celltype()

#%% read ENVI predictions
spatial_rna, rna = read_envi_predictions(spatial, rna, envi_model_rna)
spatial_atac, atac = read_envi_predictions(spatial, atac, envi_model_atac)

# %% concatenate spatial and RNA data

spatial_atac_adata = multimodal_latents_adata({'VISIUM': spatial}, {'ATAC': atac}, 'envi_latent')
spatial_rna_adata = multimodal_latents_adata({'VISIUM': spatial}, {'RNA': rna}, 'envi_latent')

# %% compute UMAP

sc.pp.pca(spatial_rna_adata, n_comps = 50) # still do PCA, since there are 512 ENVI latent dimensions
sc.pp.neighbors(spatial_rna_adata, use_rep = 'X_pca', n_neighbors = 100)
sc.tl.umap(spatial_rna_adata, min_dist = 0.3)
sc.pl.umap(spatial_rna_adata, color = ['modality'], wspace = 0.2)

sc.pp.pca(spatial_atac_adata, n_comps = 50) # still do PCA, since there are 512 ENVI latent dimensions
sc.pp.neighbors(spatial_atac_adata, use_rep = 'X_pca', n_neighbors = 100)
sc.tl.umap(spatial_atac_adata, min_dist = 0.3)
sc.pl.umap(spatial_atac_adata, color = ['modality'], wspace = 0.2)

#%% read ENVI loss metrics

log_file = os.path.join(log_output_path, "envi_atac_loss.txt")
df_metrics_atac = parse_envi_log(log_file)

log_file = os.path.join(log_output_path, "envi_rna_loss.txt")
df_metrics_rna = parse_envi_log(log_file)

df_metrics = pd.concat([
    df_metrics_atac.assign(modality="atac"),
    df_metrics_rna.assign(modality="rna")
])

losses = ['spatial', 'sc', 'cov', 'kl']

fig, ax = plt.subplots(1, len(losses), figsize=(10, 5))
for i, loss in enumerate(losses):
    sns.lineplot(data=df_metrics.reset_index(), x="index", y=loss, hue="modality", ax=ax[i])
    ax[i].set_title(loss)
    ax[i].set_xlabel("Step")
    ax[i].set_ylabel('')
plt.tight_layout()
plt.show()

#%% create SpatialData object

sdata = sd.SpatialData(
    images = {
        "histology": Image2DModel.parse(img)},
    points = {
        "spots": PointsModel.parse(
            coords[["x", "y"]].values,
            #index=coords.index
        )},
    tables = {
        "cells": spatial
    })

#%%
'''
adata = sc.read_text(os.path.join(datapath, "exprMatrix.tsv.gz")).T
meta = pd.read_csv(os.path.join(datapath, "meta.tsv"), sep="\t", index_col=0)
spots = pd.read_csv(os.path.join(datapath, "Spots.coords.tsv.gz"), sep="\t", header=None, index_col=0)
adata.obs = meta
adata.obsm["spatial"] = (spots * [1,-1]).values # flip y-axis to match layout in https://cells.ucsc.edu/?ds=brain-spatial-omics+human

atac_umap_coords = pd.read_csv(os.path.join(datapath, "ATAC_UMAP.coords.tsv.gz"), sep="\t", header=None, index_col=0).rename(columns={1: "atac_umap_1", 2: "atac_umap_2"})
rna_umap_coords = pd.read_csv(os.path.join(datapath, "RNA_UMAP.coords.tsv.gz"), sep="\t", header=None, index_col=0).rename(columns={1: "rna_umap_1", 2: "rna_umap_2"})
adata.obsm["X_umap_atac"] = atac_umap_coords.values
adata.obsm["X_umap_rna"] = rna_umap_coords.values
assert (adata.obsm["X_umap_atac"] == adata.obsm["X_umap_rna"]).flatten().all().item()

# Plot clusters using spatial coordinates
fig, ax = plt.subplots(1, 2, figsize=(8,3))
sq.pl.spatial_scatter(adata, shape=None, color='ATAC_clusters', ax=ax[0])
sq.pl.spatial_scatter(adata, shape=None, color='RNA_clusters', ax=ax[1])
plt.show()
'''
