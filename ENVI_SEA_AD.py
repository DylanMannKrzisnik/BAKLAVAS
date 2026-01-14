#%% load ENVI
import os
import subprocess

# Source .bashrc and extract CUDA-related environment variables
bashrc_path = os.path.expanduser('~/.bashrc')
if os.path.exists(bashrc_path):
    # Run bash to source .bashrc and get environment variables
    result = subprocess.run(
        ['bash', '-c', f'source {bashrc_path} && env'],
        capture_output=True,
        text=True
    )
    
    # Parse the output and update environment variables (especially CUDA-related ones)
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            # Update CUDA-related variables or merge LD_LIBRARY_PATH
            if key in ['CUDA_ROOT', 'LD_LIBRARY_PATH']:
                os.environ[key] = value

import scenvi

#%% load other libraries

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scanpy as sc
import umap.umap_ as umap

#%% load SEA-AD data
datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD'

rna = sc.read_h5ad(os.path.join(datapath, 'rna', 'middle-temporal-gyrus', 'rna_donor_H21.33.021.h5ad'))
spatial = sc.read_h5ad(os.path.join(datapath, 'merfish', 'middle-temporal-gyrus', 'merfish_section_H21.33.021.Cx26.MTG.02.007.1.04.h5ad'))

#%% scatterplots of MERFISH data

fig, ax = plt.subplots(1, 2, figsize=(10,10), sharex = True, sharey = True)

sns.scatterplot(x = spatial.obsm['spatial'][:, 0], 
                y = -spatial.obsm['spatial'][:, 1],
                legend = True, hue = spatial.obs['Subclass'], s = 12, ax = ax[0])

sns.scatterplot(x = spatial.obsm['spatial'][:, 0], 
                y = -spatial.obsm['spatial'][:, 1],
                legend = True, hue = spatial.obs['Layer annotation'], s = 12, ax = ax[1])

ax[0].axis('off'); ax[1].axis('off')
plt.suptitle("MERFISH Data")
plt.show()

#%% initialize ENVI model
rna.X = rna.X.toarray()
rna.layers['log'] = np.log1p(rna.X)

rna.obs['cell_type'] = rna.obs['Subclass']
spatial.obs['cell_type'] = spatial.obs['Subclass']

envi_model = scenvi.ENVI(spatial_data = spatial, sc_data = rna, covet_batch_size = 256)

#%% train ENVI model
envi_model.train()
envi_model.impute_genes()
envi_model.infer_niche_covet()
envi_model.infer_niche_celltype()

# %% read ENVI predictions
spatial.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
spatial.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
spatial.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
spatial.uns['COVET_genes'] =  envi_model.CovGenes
spatial.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']
spatial.obsm['cell_type_niche'] = envi_model.spatial_data.obsm['cell_type_niche']

rna.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
rna.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
rna.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
rna.uns['COVET_genes'] =  envi_model.CovGenes
rna.obsm['cell_type_niche'] = envi_model.sc_data.obsm['cell_type_niche']

# %% concatenate spatial and RNA data

spatial_rna_obs = pd.concat([
    spatial.obs.assign(modality = 'MERFISH'),
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

# %% compute UMAP
sc.pp.pca(spatial_rna_adata, n_comps = 50) # still do PCA, since there are 512 ENVI latent dimensions
sc.pp.neighbors(spatial_rna_adata, use_rep = 'X_pca', n_neighbors = 100)
sc.tl.umap(spatial_rna_adata, min_dist = 0.3)
sc.pl.umap(spatial_rna_adata, color = ['modality', 'cell_type', 'Layer annotation'], wspace = 0.2)

# %% compute cell density embedding
sc.tl.embedding_density(spatial_rna_adata, groupby = 'modality')
sc.pl.embedding_density(spatial_rna_adata, key = 'umap_density_modality')
# %%