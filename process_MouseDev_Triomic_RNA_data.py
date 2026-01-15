#%% import libraries
import os
from glob import glob
import numpy as np
import pandas as pd
import tarfile
import re
from scipy.sparse import csr_matrix
import anndata as ad
from tqdm import tqdm

#%% set paths to data and identify RNA data files from tar file
datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/MouseDev_Spatial_Triomic'
tarfilename = 'GSE308623.tar'
tarpath = os.path.join(datapath, tarfilename)

with tarfile.open(tarpath, "r:*") as tar:
    filenames = tar.getnames()

developmental_rna_pattern = r'_P\d+S\d+_RNA_matrix\.csv\.gz$'
developmental_rna_files = sorted(f for f in filenames if re.search(developmental_rna_pattern, f))

#%% process RNA data

if os.path.exists(os.path.join(datapath, 'rna_adata.h5ad')):
    print('RNA data already processed, loading from file')
    rna_adata = ad.read_h5ad(os.path.join(datapath, 'rna_adata.h5ad'))
else:
    ## define what is typical length of barcode
    barcode_length = len('AACACGGTAACAACCA-1')

    rna_adatas = []
    pbar = tqdm(developmental_rna_files)
    for file in pbar:

        sample_name = file.strip('.csv.gz').split('_')[2]
        pbar.set_description(f'Processing sample {sample_name}')

        with tarfile.open(tarpath, "r:*") as tar:
            f = tar.extractfile(file)
            rna_df = pd.read_csv(f, compression='gzip', index_col=0)

        ## check whether barcodes in index or columns
        if (rna_df.index.str.len().nunique() == 1) and (rna_df.index.str.len().unique()[0].item() == barcode_length):
            pass
        elif (rna_df.columns.str.len().nunique() == 1) and (rna_df.columns.str.len().unique()[0].item() == barcode_length):
            rna_df = rna_df.T
        else:
            raise ValueError(f'Barcodes in {file} are not in the same format')

        ## format obs
        rna_obs = rna_df.index.to_frame()
        rna_obs.columns = ['barcode']
        rna_obs = rna_obs.assign(sample_name=sample_name)
        rna_obs.index = rna_obs.apply(lambda x: x['barcode'] + '_' + x['sample_name'], axis=1)

        ## format var
        rna_var = rna_df.columns.to_frame()
        rna_var.columns = ['gene_id']

        rna_adata = ad.AnnData(
            X=csr_matrix(rna_df.values),
            obs=rna_obs,
            var=rna_var
            )

        rna_adatas.append(rna_adata)

    rna_adata = ad.concat(rna_adatas, axis=0, join='inner')
    rna_adata.write_h5ad(os.path.join(datapath, 'rna_adata.h5ad'))

#%% perform basic processing steps
import scanpy as sc

sc.pp.filter_cells(rna_adata, min_genes=100)
sc.pp.filter_genes(rna_adata, min_cells=3)

sc.pp.normalize_total(rna_adata, target_sum=1e4)
sc.pp.log1p(rna_adata)
sc.pp.highly_variable_genes(rna_adata, n_top_genes=2000)
#rna_adata = rna_adata[:, rna_adata.var['highly_variable']].copy()

sc.tl.pca(rna_adata, svd_solver='arpack', use_highly_variable=True)
sc.external.pp.harmony_integrate(rna_adata, 'sample_name')

sc.pp.neighbors(rna_adata, use_rep='X_pca_harmony')
sc.tl.umap(rna_adata, min_dist=0.1)
sc.pl.umap(rna_adata, color=['sample_name'])


#%% load spatial data

spatial_tar = 'GSE308526.tar'
spatial_tarpath = os.path.join(datapath, spatial_tar)

with tarfile.open(spatial_tarpath, "r:*") as tar:
    filenames = tar.getnames()
