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

datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/human"

#%%

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
import scenvi
from scipy.sparse import issparse

rna.X = rna.raw.X

if issparse(rna.X):
    rna.X = rna.X.toarray()
if issparse(atac.X):
    atac.X = atac.X.toarray()
if issparse(spatial.X):
    spatial.X = spatial.X.toarray()

envi_model_rna = scenvi.ENVI(spatial_data = spatial, sc_data = rna, covet_batch_size = 256)
envi_model_atac = scenvi.ENVI(spatial_data = spatial, sc_data = atac, covet_batch_size = 256)

#%% define logger and trainer
import sys
import os
from datetime import datetime
import pandas as pd
import re

# 1. Define the Tee class to handle tqdm and stdout simultaneously
class Envilogger(object):
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        # Write to notebook as-is (keeps the progress bar moving)
        self.stderr.write(data)
        
        # Write to file: replace carriage returns with newlines to see loss history
        if data.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cleaned_data = data.replace('\r', '\n').strip()
            # Avoid writing duplicate empty newlines
            if cleaned_data:
                self.file.write(f"[{timestamp}] {cleaned_data}\n")
                self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.stderr.flush()
        self.file.flush()

# 2. Configuration and Directory Setup
def envi_train_with_logger(envi_model, output_path, training_steps=10000):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. Execution with Redirection
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = Envilogger(output_path)

    try:
        sys.stdout = logger
        sys.stderr = logger
        
        print(f"--- Starting ENVI Training Session: {datetime.now()} ---")
        
        # Only train the model - do NOT call impute_genes() or infer_niche_covet()
        envi_model.train(training_steps=training_steps)        
        print(f"--- Training Complete: {datetime.now()} ---")

    finally:
        # 4. Critical: Restore streams even if the code crashes
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.file.close()
        print(f"\n[Success] Full log saved to: {output_path}")

        return envi_model


def parse_envi_log(file_path):
    # Regex to match the timestamp and the four specific loss metrics
    # It looks for patterns like spatial: -1.264e+00
    pattern = re.compile(
        r"\[(?P<timestamp>.*?)\]\s+"
        r"spatial:\s+(?P<spatial>[-+0-9.e]+)\s+"
        r"sc:\s+(?P<sc>[-+0-9.e]+)\s+"
        r"cov:\s+(?P<cov>[-+0-9.e]+)\s+"
        r"kl:\s+(?P<kl>[-+0-9.e]+):"
    )

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Convert the extracted strings to float/datetime
                row = match.groupdict()
                row['timestamp'] = pd.to_datetime(row['timestamp'])
                row['spatial'] = float(row['spatial'])
                row['sc'] = float(row['sc'])
                row['cov'] = float(row['cov'])
                row['kl'] = float(row['kl'])
                data.append(row)

    # Create DataFrame and remove duplicates (tqdm logs the same step twice sometimes)
    df = pd.DataFrame(data).drop_duplicates(subset=['spatial', 'sc', 'cov', 'kl'])
    return df.reset_index(drop=True)

def read_envi_predictions(st_dat, sc_dat, envi_model):
    st_dat.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
    st_dat.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
    st_dat.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
    st_dat.uns['COVET_genes'] =  envi_model.CovGenes
    st_dat.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']
    if 'cell_type_niche' in envi_model.spatial_data.obsm:
        st_dat.obsm['cell_type_niche'] = envi_model.spatial_data.obsm['cell_type_niche']

    sc_dat.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
    sc_dat.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
    sc_dat.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
    sc_dat.uns['COVET_genes'] =  envi_model.CovGenes
    if 'cell_type_niche' in envi_model.sc_data.obsm:
        sc_dat.obsm['cell_type_niche'] = envi_model.sc_data.obsm['cell_type_niche']

    return st_dat, sc_dat

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

## RNA data
spatial_rna_obs = pd.concat([
    spatial.obs.assign(modality = 'VISIUM'),
    rna.obs.assign(modality = 'RNA')
    ], axis = 0)

spatial_rna_envi = np.concatenate([
    spatial.obsm['envi_latent'],
    rna.obsm['envi_latent']
    ], axis = 0)

spatial_rna_adata = sc.AnnData(
    X = spatial_rna_envi,
    obs = spatial_rna_obs
)

## ATAC data
spatial_atac_obs = pd.concat([
    spatial.obs.assign(modality = 'VISIUM'),
    atac.obs.assign(modality = 'ATAC')
    ], axis = 0)

spatial_atac_envi = np.concatenate([
    spatial.obsm['envi_latent'],
    atac.obsm['envi_latent']
    ], axis = 0)

spatial_atac_adata = sc.AnnData(
    X = spatial_atac_envi,
    obs = spatial_atac_obs
)

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
import seaborn as sns

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

#%%

