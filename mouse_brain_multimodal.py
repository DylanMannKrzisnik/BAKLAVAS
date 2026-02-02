#!/usr/bin/env python
# coding: utf-8

# Notebook copied from github repo: `nichecompass/docs/tutorials/notebooks/mouse_brain_multimodal.ipynb`

# # Mouse Brain Multimodal Tutorial

# - **Creator**: Sebastian Birk (<sebastian.birk@helmholtz-munich.de>).
# - **Affiliation:** Helmholtz Munich, Institute of AI for Health (AIH), Talavera-López Lab
# - **Date of Creation:** 18.05.2023
# - **Date of Last Modification:** 21.08.2024

# In this tutorial we apply NicheCompass to a single multimodal sample (postnatal day 22 coronal section) of the spatial ATAC-RNA-seq mouse brain dataset from [Zhang, D. et al. Spatial epigenome–transcriptome co-profiling of mammalian tissues. Nature 1–10 (2023)](https://www.nature.com/articles/s41586-023-05795-1).
# 
# The sample has:
# - 9215 observations at spot resolution with spot rna cluster and atac cluster annotations
# - 22,914 probed genes
# - 121,068 called peaks

# - Check the repository [README.md](https://github.com/sebastianbirk/nichecompass#installation) for NicheCompass installation instructions.
# - The data for this tutorial can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1l9W0MDVZ451k1L7s6GGH4ONH4tEK4EKj). It has to be stored under ```<repository_root>/data/spatial_omics/```.
#     - spatial_atac_rna_seq_mouse_brain_atac.h5ad
#     - spatial_atac_rna_seq_mouse_brain.h5ad
#     - spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv
# - A pretrained model to run only the analysis can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1z2DQHV9hG22B5OSWox8U3usf_8LKZGGH). It has to be stored under ```<repository_root>/artifacts/multimodal/<timestamp>/model/```.
#     - ```<timestamp>```: 22082024_142839

# ## 1. Setup

#%% 1.1 Import Libraries

import os
import random
import warnings
from datetime import datetime
from typing import Optional

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mygene as mg
import anndata as ad
from muon import MuData

from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                add_multimodal_mask_to_adata,
                                create_new_color_dict,
                                compute_communication_gp_network,
                                visualize_communication_gp_network,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps_v2,
                                get_gene_annotations,
                                generate_enriched_gp_info_plots,
                                generate_multimodal_mapping_dict,
                                get_unique_genes_from_gp_dict)

#%% Import custom NicheCompass and DataAligner classes

import sys

# Allow importing `nichecompass_utils.py` from repo root when running this script
BAKLAVA_ROOT = "/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA"
sys.path.append(str(BAKLAVA_ROOT))
from nichecompass_utils import CustomNicheCompass
from data_aligner import DataAligner

#%% 1.2 Define Parameters


### Dataset ###
dataset = "spatial_atac_rna_seq_mouse_brain"
species = "mouse"
spatial_key = "spatial"
n_neighbors = 4
n_sampled_neighbors = 4
filter_genes = True
n_svg = 3000
n_svp = 15000
filter_peaks = True
min_cell_peak_thresh_ratio = 0.005 # 0.05%
min_cell_gene_thresh_ratio = 0.005 # 0.05%

### Model ###
# AnnData keys
counts_key = "counts"
adj_key = "spatial_connectivities"
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

# Architecture
active_gp_thresh_ratio = 0.01
conv_layer_encoder = "gatv2conv" # default is "gatv2conv", change to "gcnconv" if not enough compute and memory

# Trainer
n_epochs = 400
n_epochs_all_gps = 25
lr = 0.001
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_chrom_access_recon = 300.
lambda_l1_masked = 0. # prior GP  regularization
lambda_l1_addon = 30. # de novo GP regularization
edge_batch_size = 64 # increase if more memory available or decrease to save memory
use_cuda_if_available = True

### Analysis ###
cell_type_key = "cell_type"
latent_leiden_resolution = 0.6
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 30
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"


#%% 1.3 Run Notebook Setup

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# Notebook-style pretty display (safe fallback for script execution)
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)

# Get time of notebook execution for timestamping saved artifacts
now = datetime.now()
current_timestamp = now.strftime("%d%m%Y_%H%M%S")


#%% 1.4 Configure Paths

# Define paths
datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse"
outpath = "/home/mcb/users/dmannk/BAKLAVA_base/outputs/nichecompass_mouse_brain_multimodal"
os.makedirs(datapath, exist_ok=True)
os.makedirs(outpath, exist_ok=True)

ga_data_folder_path = f"{datapath}/gene_annotations"
gp_data_folder_path = f"{datapath}/gene_programs"
so_data_folder_path = f"{datapath}/spatial_omics"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
nichenet_ligand_target_matrix_file_path = f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"
mebocost_enzyme_sensor_interactions_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
collectri_tf_network_file_path = f"{gp_data_folder_path}/collectri_tf_network_{species}.csv"
marker_gp_folder_path = f"{gp_data_folder_path}/marker_gps"
gene_orthologs_mapping_file_path = f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
gtf_file_path = f"{ga_data_folder_path}/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"
artifacts_folder_path = f"{outpath}/artifacts"
model_folder_path = f"{artifacts_folder_path}/multimodal/{current_timestamp}/model"
figure_folder_path = f"{artifacts_folder_path}/multimodal/{current_timestamp}/figures"


#%% 1.5 Create Directories

os.makedirs(model_folder_path, exist_ok=True)
os.makedirs(figure_folder_path, exist_ok=True)
os.makedirs(so_data_folder_path, exist_ok=True)
os.makedirs(gp_data_folder_path, exist_ok=True)
os.makedirs(ga_data_folder_path, exist_ok=True)


#%% 1.6 Download Files (Optional)
# You can skip this part if you have downloaded the files mentioned above manually, or you are using your own data.

if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain_atac.h5ad')):
    gdown.download("https://drive.google.com/file/d/1WH_9PYV_AEcLd5QVNILig-_gfS-EdRVR', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain_atac.h5ad")
if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain.h5ad')):
    gdown.download("https://drive.google.com/file/d/1NpRynEDnGnxab6sHJy4AeitSKnwfX2Mi', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain.h5ad")
if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv')):
    gdown.download("https://drive.google.com/file/d/1w3eorJojnndhmeTicbin7eRjR8Pwj7O-', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv")

#%% 2. MODEL PREPARATION

# - NicheCompass expects a prior GP mask as input, which it will use to make its latent feature space interpretable (through linear masked decoders). 
# - The user can provide a custom GP mask to NicheCompass based on the biological question of interest.
# - As a default, we create a GP mask based on four databases of prior knowledge of inter- and intracellular interaction pathways:
#     - OmniPath (Ligand-Receptor GPs)
#     - MEBOCOST (Enzyme-Sensor GPs)
#     - CollecTRI (Transcriptional Regulation GPs)
#     - NicheNet (Combined Interaction GPs)


#%% 2.1 Create Prior Knowledge Gene Program (GP) Mask


# Retrieve OmniPath GPs (source: ligand genes; target: receptor genes)
omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    species=species,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=omnipath_lr_network_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True,
    gp_gene_count_distributions_save_path=f"{figure_folder_path}" \
                                           "/omnipath_gp_gene_count_distributions.svg")


# Display example OmniPath GP
omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]
print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")

#%% Retrieve NicheNet GPs (source: ligand genes; target: receptor genes, target genes)

save_load_kwargs = {
    'lr_network_file_path': nichenet_lr_network_file_path,
    'ligand_target_matrix_file_path': nichenet_ligand_target_matrix_file_path,
    'gene_orthologs_mapping_file_path': gene_orthologs_mapping_file_path,
}

## check if files exist
if os.path.exists(nichenet_lr_network_file_path):
    save_load_kwargs.update({
        'load_from_disk': True,
        'save_to_disk': False,
    })
else:
    save_load_kwargs.update({
        'save_to_disk': True,
        'load_from_disk': False,
    })

## extract GP dict
nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,
    version="v2",
    keep_target_genes_ratio=1.,
    max_n_target_genes_per_gp=250,
    plot_gp_gene_count_distributions=True,
    **save_load_kwargs)


# Display example NicheNet GP
nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]
print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")


#%% Retrieve MEBOCOST GPs (source: enzyme genes; target: sensor genes)

mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,
    species=species,
    plot_gp_gene_count_distributions=True)

# Display example MEBOCOST GP
mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]
print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")


#%% Retrieve CollecTRI GPs (source: -; target: transcription factor genes, target genes)

collectri_gp_dict = extract_gp_dict_from_collectri_tf_network(
        species=species,
        tf_network_file_path=collectri_tf_network_file_path,
        load_from_disk=False,
        save_to_disk=True,
        plot_gp_gene_count_distributions=True)


# Display example CollecTRI GP
collectri_gp_names = list(collectri_gp_dict.keys())
random.shuffle(collectri_gp_names)
collectri_gp_name = collectri_gp_names[0]
print(f"{collectri_gp_name}: {collectri_gp_dict[collectri_gp_name]}")


#%% Filter and combine GPs

gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict, collectri_gp_dict]
combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
    gp_dicts,
    verbose=True)

print(f"Number of gene programs after filtering and combining: "
      f"{len(combined_gp_dict)}.")


#%% 2.2 Load source data

# - NicheCompass expects a precomputed spatial adjacency matrix stored in 'adata.obsp[adj_key]'.
# - The user can customize the spatial neighbor graph construction based on the biological question of interest.
# - In the multimodal setting, we will provide one adata object per modality to NicheCompass.


# Read data
adata = sc.read_h5ad(
        f"{so_data_folder_path}/{dataset}.h5ad")
adata_atac = sc.read_h5ad(
        f"{so_data_folder_path}/{dataset}_atac.h5ad")

# Load and add cell type annotations
cell_type_df = pd.read_csv(f"{so_data_folder_path}/{dataset}_cell_type_annotations.csv", index_col=0)
cell_type_df.rename({"predicted.celltype": cell_type_key}, axis=1, inplace=True)
cell_type_df.drop("ATAC_clusters", axis=1, inplace=True)
adata.obs = adata.obs.merge(cell_type_df, left_index=True, right_index=True, how="left")


#%% Compute spatial neighborhood

sq.gr.spatial_neighbors(adata,
                        coord_type="generic",
                        spatial_key=spatial_key,
                        n_neighs=n_neighbors)

# Make adjacency matrix symmetric
adata.obsp[adj_key] = (
    adata.obsp[adj_key].maximum(
        adata.obsp[adj_key].T))


#%% 2.3 Filter Genes & Peaks

if filter_genes:
    print("Filtering genes...")
    # Filter genes and only keep ligand, receptor, enzyme, sensor, and
    # the 'n_svg' spatially variable genes
    gp_dict_genes = get_unique_genes_from_gp_dict(
        gp_dict=combined_gp_dict,
            retrieved_gene_entities=["sources", "targets"])
    print(f"Starting with {len(adata.var_names)} genes.")
    min_cells = int(adata.shape[0] * min_cell_gene_thresh_ratio)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Keeping {len(adata.var_names)} genes after filtering genes with "
          f"counts in less than {int(adata.shape[0] * min_cell_gene_thresh_ratio)} cells.")
    
    # Identify spatially variable genes
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    svg_genes = adata.uns["moranI"].index[:n_svg].tolist()
    adata.var["spatially_variable"] = adata.var_names.isin(svg_genes)
    adata = adata[:, adata.var["spatially_variable"] == True]
    print(f"Keeping {len(adata.var_names)} spatially variable genes.")
    
if filter_peaks:
    print("\nFiltering peaks...")
    print(f"Starting with {len(adata_atac.var_names)} peaks.")
    # Filter out peaks that are rarely detected to reduce GPU footprint of model
    min_cells = int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)
    sc.pp.filter_genes(adata_atac, min_cells=min_cells)
    print(f"Keeping {len(adata_atac.var_names)} peaks after filtering peaks with "
          f"counts in less than {int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)} cells.")
    
    # Filter spatially variable peaks
    adata_atac.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
    adata_atac.obsp["spatial_distances"] = adata.obsp["spatial_distances"]

    sq.gr.spatial_autocorr(adata_atac,
                           mode="moran",
                           genes=adata_atac.var_names)
    sv_peaks = adata_atac.uns["moranI"].index[:n_svp].tolist()
    adata_atac.var["spatially_variable"] = adata_atac.var_names.isin(sv_peaks)
    adata_atac = adata_atac[:, adata_atac.var["spatially_variable"] == True]
    print(f"Keeping {len(adata_atac.var_names)} peaks after filtering spatially variable "
          f"peaks.")

print("\n WARNING: genes currently not filtered by GP, 'gp_dict_genes' not used.")


#%% 2.4 Annotate Genes & Peaks

# Next we will add positional annotations to genes and peaks to be able to match spatially proximal peaks to genes.

adata, adata_atac = get_gene_annotations(
    adata=adata,
    adata_atac=adata_atac,
    gtf_file_path=gtf_file_path)


# Display gene annotations
print(adata.var[["chrom", "chromStart", "chromEnd"]])

# Display peak annotations
print(adata_atac.var[["chrom", "chromStart", "chromEnd"]])

#%% 2.7 Explore Data

cell_type_colors = create_new_color_dict(
    adata=adata,
    skip_default_colors=50,
    cat_key=cell_type_key)

print(f"Number of nodes (observations): {adata.layers['counts'].shape[0]}")
print(f"Number of gene node features: {adata.layers['counts'].shape[1]}")
print(f"Number of peak node features: {adata_atac.layers['counts'].shape[1]}")

# Visualize spot-level annotated data in physical space
sc.pl.spatial(adata,
              color=cell_type_key,
              palette=cell_type_colors,
              spot_size=spot_size)


#%% Load target data

def load_easysci_sll_data():

    target_rna = ad.read_h5ad("/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL/mouse/RNA/mouse_rna_processed.h5ad")
    target_atac = ad.read_h5ad("/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL/mouse/ATAC/mouse_atac_processed.h5ad")

    mginfo = mg.MyGeneInfo()
    results = mginfo.querymany(
        target_rna.var["gene_id_no_version"].tolist(),
        scopes="ensembl.gene",
        species="mouse",
        fields="symbol",
        as_dataframe=True,
    )
    results = results.reset_index().drop_duplicates(subset='query') # remove duplicate genes
    assert results.groupby('query')['symbol'].nunique().le(1).all(), "Multiple symbols still found for some genes"

    target_rna.var = target_rna.var.merge(results, left_on="gene_id_no_version", right_on="query", how="left")
    target_rna.var.loc[target_rna.var['symbol'].isna(), 'symbol'] = target_rna.var.loc[target_rna.var['symbol'].isna(), 'query']
    target_rna.var.set_index("symbol", inplace=True)

    ## remove duplicate genes (again)
    target_rna = target_rna[:, ~target_rna.var_names.duplicated(keep='first')]
    assert target_rna.var_names.is_unique, "Target RNA data must have unique gene names"

    target_atac.var[['chrom', 'chromStart', 'chromEnd']] = target_atac.var['peak'].str.split('-').tolist()

    return target_rna, target_atac

def load_10x_mouse_brain_ad_data(
    data_dir: Optional[str] = None,
    base_name: str = "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote",
) -> tuple:
    """
    Load processed RNA and ATAC data from 10x Multiome Mouse Brain Alzheimers AppNote.

    Expects the following files in `data_dir` (downloaded as per 10x output):
    - {base_name}_filtered_feature_bc_matrix.h5  (required)
    - {base_name}_atac_peaks.bed                  (optional, for peak coordinates in ATAC var)
    - {base_name}_atac_peak_annotation.tsv        (optional, for peak annotations in ATAC var)

    Returns
    -------
    tuple of (adata_rna, adata_atac)
        RNA and ATAC AnnData objects with shared obs (cells). Raw counts in .layers["counts"] and .X.
    """
    if data_dir is None:
        data_dir = os.path.join(BAKLAVA_ROOT, "..", "data", "10x_mouse_brain_AD")
    data_dir = os.path.abspath(data_dir)
    h5_path = os.path.join(data_dir, f"{base_name}_filtered_feature_bc_matrix.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"10x filtered feature-barcode matrix not found: {h5_path}. "
            "Download it from 10x (e.g. filtered_feature_bc_matrix.h5) into data_dir."
        )

    # Read full multiome matrix (Gene Expression + Peaks); gex_only=False keeps both modalities
    adata_full = sc.read_10x_h5(h5_path, gex_only=False)
    # 10x multiome var has 'feature_types': "Gene Expression" vs "Peaks"
    ft_col = "feature_types" if "feature_types" in adata_full.var.columns else "feature_type"
    if ft_col not in adata_full.var.columns:
        raise ValueError(
            f"Expected feature type column '{ft_col}' in 10x multiome var. "
            f"Columns: {list(adata_full.var.columns)}"
        )

    is_gex = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("gene expression")
    is_peaks = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("peaks")

    adata_rna = adata_full[:, is_gex].copy()
    adata_atac = adata_full[:, is_peaks].copy()

    # Drop the feature type column from each modality's var so we don't duplicate
    for a in (adata_rna, adata_atac):
        if ft_col in a.var.columns:
            a.var = a.var.drop(columns=[ft_col])

    # Store raw counts in .layers["counts"] for compatibility with counts_key
    adata_rna.layers["counts"] = adata_rna.X.copy()
    adata_atac.layers["counts"] = adata_atac.X.copy()

    # Optionally add ATAC peak coordinates and annotations from BED / peak_annotation.tsv
    peaks_bed_path = os.path.join(data_dir, f"{base_name}_atac_peaks.bed")
    peak_ann_path = os.path.join(data_dir, f"{base_name}_atac_peak_annotation.tsv")
    if os.path.isfile(peaks_bed_path):
        bed = pd.read_csv(
            peaks_bed_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "chromStart", "chromEnd"],
        )
        # 10x peak names are "chr1-123-456"; BED order may match var order
        peak_id_bed = bed["chrom"].astype(str) + "-" + bed["chromStart"].astype(str) + "-" + bed["chromEnd"].astype(str)
        if adata_atac.n_vars == len(peak_id_bed) and (adata_atac.var_names == peak_id_bed.values).all():
            adata_atac.var["chrom"] = bed["chrom"].values
            adata_atac.var["chromStart"] = bed["chromStart"].values
            adata_atac.var["chromEnd"] = bed["chromEnd"].values
        else:
            # Parse coordinates from peak name (chr-start-end)
            adata_atac.var["peak"] = adata_atac.var_names
            coords = adata_atac.var["peak"].str.split("-", n=2, expand=True)
            adata_atac.var["chrom"] = coords[0]
            adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
            adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""
    else:
        adata_atac.var["peak"] = adata_atac.var_names
        coords = adata_atac.var["peak"].str.split("-", n=2, expand=True)
        adata_atac.var["chrom"] = coords[0]
        adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
        adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""

    if os.path.isfile(peak_ann_path):
        ann = pd.read_csv(peak_ann_path, sep="\t")
        peak_col = "Peak" if "Peak" in ann.columns else ("peak" if "peak" in ann.columns else None)
        if peak_col is not None:
            ann_indexed = ann.set_index(peak_col)
            # Only merge columns that are not already in adata_atac.var to avoid duplicates
            extra = [c for c in ann_indexed.columns if c not in adata_atac.var.columns]
            if extra:
                adata_atac.var = adata_atac.var.merge(
                    ann_indexed[extra], left_index=True, right_index=True, how="left"
                )

    return adata_rna, adata_atac, "mm10", "10x_mouse_brain_AD"

## load target data
target_rna, target_atac, target_assembly, target_name = load_10x_mouse_brain_ad_data()

target_rna.var_names = target_rna.var_names.str.split(".").str[0]
target_rna = target_rna[:, ~target_rna.var_names.duplicated(keep='first')]

target_atac.var[['chrom', 'chromStart', 'chromEnd']] = target_atac.var['peak'].str.split(':|-').tolist()

#%% Basic feature processing

assert \
    "counts" in adata.layers and \
    "counts" in adata_atac.layers and \
    "counts" in target_rna.layers and \
    "counts" in target_atac.layers, \
    "Counts layer not found in source or target data"

## source RNA data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

## source ATAC data
sc.pp.normalize_total(adata_atac, target_sum=1e4)
sc.pp.log1p(adata_atac)

## target RNA data
sc.pp.filter_cells(target_rna, min_genes=100)
sc.pp.filter_genes(target_rna, min_cells=3)
sc.pp.normalize_total(target_rna, target_sum=1e4)
sc.pp.log1p(target_rna)

## target ATAC data
sc.pp.filter_cells(target_atac, min_genes=100)
sc.pp.filter_genes(target_atac, min_cells=3)
sc.pp.normalize_total(target_atac, target_sum=1e4)
sc.pp.log1p(target_atac)

## peform PCA and compute neighbor graph for target data
sc.pp.pca(target_rna, n_comps=50)
sc.pp.neighbors(target_rna, use_rep='X_pca', n_neighbors=100)

sc.pp.pca(target_atac, n_comps=50)
sc.pp.neighbors(target_atac, use_rep='X_pca', n_neighbors=100)

#%% Perform data alignment

source_data = MuData({"rna": adata, "atac": adata_atac})
source_name = "Spatial_ATAC_RNA"
source_assembly = "mm10"

target_data = MuData({"rna": target_rna, "atac": target_atac})

data_aligner = DataAligner(
    source_data=source_data,
    target_data=target_data,
    source_name=source_name,
    target_name=target_name,
    source_assembly=source_assembly,
    target_assembly=target_assembly,
)
data_aligner.find_gene_overlap()
data_aligner.find_peak_overlap()
data_aligner.align_features_by_overlap()

## retrieve aligned data
adata = data_aligner.source_data['rna']
adata_atac = data_aligner.source_data['atac']

target_rna = data_aligner.target_data['rna']
target_atac = data_aligner.target_data['atac']

#%% Rebuild GP + multimodal masks on aligned feature space
#
# After alignment, (re)create all masks so they are strictly tied to the final
# aligned `adata.var_names` / `adata_atac.var_names`.

# Defensive cleanup in case this cell was executed before (e.g., in notebooks)
for _k in [gp_targets_mask_key, gp_targets_categories_mask_key, gp_sources_mask_key, gp_sources_categories_mask_key]:
    if _k in adata.varm:
        del adata.varm[_k]
for _k in [gp_names_key]:
    if _k in adata.uns:
        del adata.uns[_k]

# Add the GP dictionary as binary masks to the (aligned) RNA adata
add_gps_from_gp_dict_to_adata(
    gp_dict=combined_gp_dict,
    adata=adata,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=0,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None,
    plot_gp_gene_count_distributions=True
    )

## Based on spatial proximity to the genes in the GP mask, add chromatin accessibility masks
gene_peak_mapping_dict = generate_multimodal_mapping_dict(
    adata=adata,
    adata_atac=adata_atac
    )

## Add chromatin accessibility masks
filter_peaks_based_on_genes = True    # If ´True´, filter ´adata_atac´ to only keep peaks that are mapped to genes in ´gene_peak_mapping_dict´.
adata, adata_atac = add_multimodal_mask_to_adata(
    adata=adata,
    adata_atac=adata_atac,
    gene_peak_mapping_dict=gene_peak_mapping_dict,
    filter_peaks_based_on_genes=filter_peaks_based_on_genes
    )

if filter_peaks_based_on_genes:
    print(f"Keeping {adata_atac.n_vars} peaks after filtering peaks with "
        "no matching genes in gp mask.")

    remaining_peaks = target_atac.var_names.isin(adata_atac.var_names)
    target_atac = target_atac[:, remaining_peaks]

assert adata.var_names.equals(target_rna.var_names), "RNA data must have the same gene names"
assert adata_atac.var_names.equals(target_atac.var_names), "ATAC data must have the same peak names"

#%% Copy annotations and set spatial connectivities for target data

target_rna, target_atac = DataAligner.copy_annotations_to_target(
    source_rna=adata,
    source_atac=adata_atac,
    target_rna=target_rna,
    target_atac=target_atac
)
target_rna, target_atac = DataAligner.set_target_spatial_connectivities(
    target_rna=target_rna,
    target_atac=target_atac,
    adj_type="knn"
)

#%% Convert non-string columns in var to string representation
# Fix non-string columns in var that can't be saved to H5AD
def fix_var_for_h5ad(adata):
    """Convert non-string columns in var to string representation."""
    
    for col in list(adata.var.columns):  # Use list() to avoid modification during iteration
        try:
            # For object dtype columns, ensure ALL values are strings
            if adata.var[col].dtype == 'object':
                # Convert all values to strings, handling None/NaN and mixed types
                def safe_str_convert(x):
                    if pd.isna(x) or x is None:
                        return ''
                    elif isinstance(x, str):
                        return x
                    elif isinstance(x, (list, tuple, dict, np.ndarray)):
                        return str(x)
                    elif isinstance(x, (int, float, bool)):
                        return str(x)
                    else:
                        # Try to convert anything else to string
                        try:
                            return str(x)
                        except:
                            return ''
                
                # Apply conversion to all values
                adata.var[col] = adata.var[col].apply(safe_str_convert)
                # Ensure the dtype is object with string values
                adata.var[col] = adata.var[col].astype(str)
            
            # Also check for categorical dtypes - convert to string
            elif hasattr(adata.var[col].dtype, 'categories'):
                # Categorical dtype - convert to string
                adata.var[col] = adata.var[col].astype(str)
                
        except Exception as e:
            print(f"Warning: Could not fix column '{col}', dropping it: {e}")
            import traceback
            traceback.print_exc()
            adata.var = adata.var.drop(columns=[col])
    
    return adata

# First, identify problematic columns
def diagnose_var_columns(adata, name="adata"):
    """Diagnose which columns might cause issues."""
    print(f"\nDiagnosing {name}.var columns:")
    for col in adata.var.columns:
        dtype = adata.var[col].dtype
        print(f"  {col}: dtype={dtype}")
        if dtype == 'object':
            sample_vals = adata.var[col].dropna().head(3)
            for idx, val in sample_vals.items():
                print(f"    Sample value type: {type(val)}, value: {val}")

# Fix both RNA and ATAC var tables
print("Fixing var tables for H5AD compatibility...")

diagnose_var_columns(target_rna, "RNA")
diagnose_var_columns(target_atac, "ATAC")

target_rna = fix_var_for_h5ad(target_rna)
target_atac = fix_var_for_h5ad(target_atac)

# try: target_rna.write_h5ad(os.path.join(model_folder_path, 'target_rna.h5ad')); print(os.path.join(model_folder_path, 'target_rna.h5ad'))

#%% Create shallow references to pseudocounts in new layers

adata.layers["pseudocounts"] = adata.X
adata_atac.layers["pseudocounts"] = adata_atac.X

target_rna.layers["pseudocounts"] = target_rna.X
target_atac.layers["pseudocounts"] = target_atac.X

#%% Initialize model

model = CustomNicheCompass(
    adata,
    adata_atac,
    counts_key=counts_key,
    adj_key=adj_key,
    gp_names_key=gp_names_key,
    active_gp_names_key=active_gp_names_key,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    active_gp_thresh_ratio=active_gp_thresh_ratio,
    latent_key=latent_key,
    conv_layer_encoder=conv_layer_encoder,
    encoder_input_key="pseudocounts",
    multimodal_layer_series=True,
    multimodal_embedding_size=None,
)


#%% Train model

# Set up MLflow experiment for logging
mlflow_experiment_name = f"nichecompass_mouse_brain_multimodal"

# Get or create experiment
mlflow_experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
if mlflow_experiment is None:
    # Create new experiment
    mlflow_experiment_id = mlflow.create_experiment(mlflow_experiment_name)
    print(f"Created new MLflow experiment: {mlflow_experiment_name} (ID: {mlflow_experiment_id})")
else:
    mlflow_experiment_id = mlflow_experiment.experiment_id
    print(f"Using existing MLflow experiment: {mlflow_experiment_name} (ID: {mlflow_experiment_id})")

# Set the active experiment
mlflow.set_experiment(experiment_name=mlflow_experiment_name)

# Start MLflow run with timestamp-based name
with mlflow.start_run(run_name=current_timestamp):

    ## Log source and target dataset names in MLflow
    mlflow.log_param("source_dataset_name", source_name)
    mlflow.log_param("target_dataset_name", target_name)

    ## Train model
    model.train(n_epochs=n_epochs,
                n_epochs_all_gps=n_epochs_all_gps,
                lr=lr,
                lambda_edge_recon=lambda_edge_recon,
                lambda_gene_expr_recon=lambda_gene_expr_recon,
                lambda_chrom_access_recon=lambda_chrom_access_recon,
                lambda_l1_masked=lambda_l1_masked,
                lambda_l1_addon=lambda_l1_addon,
                lambda_multimodal_contrastive_loss=100.0,
                edge_batch_size=edge_batch_size,
                use_cuda_if_available=use_cuda_if_available,
                n_sampled_neighbors=n_sampled_neighbors,
                multimodal_contrastive_anneal=False,
                target_adata=target_rna,
                target_adata_atac=target_atac,
                target_holdout_frac=0.1,
                target_holdout_n=2000,
                target_holdout_seed=0,
                target_paired_data=True,
                target_encoder_input_key='pseudocounts',
                target_counts_key=counts_key,
                log_target_multimodal_contrastive=True,
                use_early_stopping=False,
                verbose=False,
                mlflow_experiment_id=mlflow_experiment_id
            )


#%% Compute latent neighbor graph & UMAP embedding
sc.pp.neighbors(model.adata,
                use_rep=latent_key,
                key_added=latent_key)

sc.tl.umap(model.adata,
           neighbors_key=latent_key)


#%% Save trained model (and data?)

model.save(dir_path=model_folder_path,
           overwrite=True,
           save_adata=True,
           adata_file_name="adata.h5ad",
           save_adata_atac=True,
           adata_atac_file_name=f"adata_atac.h5ad"
          )


#%% 4. ANALYSIS FUNCTIONS

def benchmark_clip_embeddings(clip_embeddings_adata):
    from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

    clip_embeddings_adata.obsm['X'] = clip_embeddings_adata.X

    bm = Benchmarker(
        clip_embeddings_adata,
        batch_key="modality",
        label_key="RNA_clusters",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X", "X_pca"],
        #pre_integrated_embedding_obsm_key="X_gps",
        n_jobs=6,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False)

def compute_and_store_latent_representation(
    model: CustomNicheCompass,
    latent_key: str,
    counts_key: Optional[str] = None,
    adj_key: Optional[str] = None,
    only_active_gps: bool = True,
    paired_data: bool = True
):
    """
    Compute latent representation for a model and store it in adata.obsm.
    
    This is necessary for models loaded with new data (e.g., target data) where
    the latent representation hasn't been computed yet.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    latent_key : str
        Key to store latent representation in adata.obsm
    counts_key : Optional[str]
        Key for counts in adata.layers (uses model's default if None)
    adj_key : Optional[str]
        Key for adjacency matrix in adata.obsp (uses model's default if None)
    only_active_gps : bool
        Whether to return only active gene programs
    paired_data : bool
        Whether RNA and ATAC data are paired (same cells)
    """
    if counts_key is None:
        counts_key = model.counts_key_
    if adj_key is None:
        adj_key = model.adj_key_
    
    # Compute latent representation
    z, _ = model.get_latent_representation(
        adata=model.adata,
        adata_atac=model.adata_atac if hasattr(model, 'adata_atac') else None,
        paired_data=paired_data,
        counts_key=counts_key,
        adj_key=adj_key,
        cat_covariates_keys=None,
        only_active_gps=only_active_gps,
        return_mu_std=True,
        node_batch_size=getattr(model, 'node_batch_size_', 64),
    )
    
    # Store in adata.obsm
    model.adata.obsm[latent_key] = z
    
    print(f"Computed and stored latent representation with shape {z.shape} "
          f"in model.adata.obsm['{latent_key}']")


def visualize_cell_types_in_latent_and_physical_space(
    model: CustomNicheCompass,
    samples: list,
    cell_type_key: str,
    sample_key: str,
    latent_key: str,
    figure_folder_path: str,
    spot_size: int = 30,
    groups: Optional[list] = None,
    save_fig: bool = True,
    file_suffix: str = "",
    skip_default_colors: int = 50
):
    """
    Visualize cell types in latent (UMAP) and physical (spatial) space.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    samples : list
        List of sample names to visualize
    cell_type_key : str
        Key in adata.obs for cell type annotations
    sample_key : str
        Key in adata.obs for sample annotations
    latent_key : str
        Key for latent representation in adata.obsm
    figure_folder_path : str
        Path to save figures
    spot_size : int
        Size of spots in spatial plots
    groups : Optional[list]
        Specific groups to highlight (None for all)
    save_fig : bool
        Whether to save the figure
    file_suffix : str
        Suffix to add to filename (e.g., "_source" or "_target")
    skip_default_colors : int
        Number of default colors to skip when creating color dict
    """
    cell_type_colors = create_new_color_dict(
        adata=model.adata,
        skip_default_colors=skip_default_colors,
        cat_key=cell_type_key)
    
    file_path = f"{figure_folder_path}/cell_types_latent_physical_space{file_suffix}.svg"
    
    fig = plt.figure(figsize=(12, 14))
    title = fig.suptitle(t="Cell Types in Latent and Physical Space",
                         y=0.96,
                         x=0.55,
                         fontsize=20)
    spec1 = gridspec.GridSpec(ncols=1,
                              nrows=2,
                              width_ratios=[1],
                              height_ratios=[3, 2])
    spec2 = gridspec.GridSpec(ncols=len(samples),
                              nrows=2,
                              width_ratios=[1] * len(samples),
                              height_ratios=[3, 2])
    axs = []
    axs.append(fig.add_subplot(spec1[0]))
    sc.pl.umap(adata=model.adata,
               color=[cell_type_key],
               groups=groups,
               palette=cell_type_colors,
               title=f"Cell Types in Latent Space",
               ax=axs[0],
               size=40,
               show=False)
    for idx, sample in enumerate(samples):
        axs.append(fig.add_subplot(spec2[len(samples) + idx]))
        sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                      color=[cell_type_key],
                      groups=groups,
                      palette=cell_type_colors,
                      spot_size=spot_size,
                      title=f"Cell Types in Physical Space \n"
                            f"(Sample: {sample})",
                      legend_loc=None,
                      ax=axs[idx+1],
                      show=False)
    
    # Create and position shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles,
                     labels,
                     loc="center left",
                     bbox_to_anchor=(0.98, 0.5))
    axs[0].get_legend().remove()
    
    # Adjust, save and display plot
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    if save_fig:
        fig.savefig(file_path,
                    bbox_extra_artists=(lgd, title),
                    bbox_inches="tight")
    plt.show()
    
    return cell_type_colors


def identify_niches(
    model: CustomNicheCompass,
    samples: list,
    sample_key: str,
    latent_key: str,
    latent_cluster_key: str,
    latent_leiden_resolution: float,
    figure_folder_path: str,
    spot_size: int = 30,
    groups: Optional[list] = None,
    save_fig: bool = True,
    file_suffix: str = ""
):
    """
    Identify niches using Leiden clustering and visualize in latent and physical space.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    samples : list
        List of sample names to visualize
    sample_key : str
        Key in adata.obs for sample annotations
    latent_key : str
        Key for latent representation in adata.obsm
    latent_cluster_key : str
        Key to store Leiden cluster results in adata.obs
    latent_leiden_resolution : float
        Resolution parameter for Leiden clustering
    figure_folder_path : str
        Path to save figures
    spot_size : int
        Size of spots in spatial plots
    groups : Optional[list]
        Specific groups to highlight (None for all)
    save_fig : bool
        Whether to save the figure
    file_suffix : str
        Suffix to add to filename
        
    Returns
    -------
    latent_cluster_colors : dict
        Color dictionary for clusters
    """
    # Compute latent Leiden clustering
    sc.tl.leiden(adata=model.adata,
                 resolution=latent_leiden_resolution,
                 key_added=latent_cluster_key,
                 neighbors_key=latent_key)
    
    latent_cluster_colors = create_new_color_dict(
        adata=model.adata,
        cat_key=latent_cluster_key)
    
    file_path = f"{figure_folder_path}/res_{latent_leiden_resolution}_niches_latent_physical_space{file_suffix}.svg"
    
    fig = plt.figure(figsize=(12, 14))
    title = fig.suptitle(t=f"NicheCompass Niches "
                            "in Latent and Physical Space",
                         y=0.96,
                         x=0.55,
                         fontsize=20)
    spec1 = gridspec.GridSpec(ncols=1,
                              nrows=2,
                              width_ratios=[1],
                              height_ratios=[3, 2])
    spec2 = gridspec.GridSpec(ncols=len(samples),
                              nrows=2,
                              width_ratios=[1] * len(samples),
                              height_ratios=[3, 2])
    axs = []
    axs.append(fig.add_subplot(spec1[0]))
    sc.pl.umap(adata=model.adata,
               color=[latent_cluster_key],
               groups=groups,
               palette=latent_cluster_colors,
               title=f"Niches in Latent Space",
               ax=axs[0],
               size=40,
               show=False)
    for idx, sample in enumerate(samples):
        axs.append(fig.add_subplot(spec2[len(samples) + idx]))
        sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                      color=[latent_cluster_key],
                      groups=groups,
                      palette=latent_cluster_colors,
                      spot_size=spot_size,
                      title=f"Niches in Physical Space \n"
                            f"(Sample: {sample})",
                      legend_loc=None,
                      ax=axs[idx+1],
                      show=False)
    
    # Create and position shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    lgd = fig.legend(handles,
                     labels,
                     loc="center left",
                     bbox_to_anchor=(0.98, 0.5))
    axs[0].get_legend().remove()
    
    # Adjust, save and display plot
    plt.subplots_adjust(wspace=0.2, hspace=0.25)
    if save_fig:
        fig.savefig(file_path,
                    bbox_extra_artists=(lgd, title),
                    bbox_inches="tight")
    plt.show()
    
    return latent_cluster_colors


def characterize_niche_composition(
    model: CustomNicheCompass,
    latent_cluster_key: str,
    cell_type_key: str,
    latent_leiden_resolution: float,
    figure_folder_path: str,
    save_fig: bool = True,
    file_suffix: str = ""
):
    """
    Characterize niche composition by cell type.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    latent_cluster_key : str
        Key in adata.obs for niche/cluster annotations
    cell_type_key : str
        Key in adata.obs for cell type annotations
    latent_leiden_resolution : float
        Resolution parameter used for Leiden clustering
    figure_folder_path : str
        Path to save figures
    save_fig : bool
        Whether to save the figure
    file_suffix : str
        Suffix to add to filename
    """
    file_path = f"{figure_folder_path}/res_{latent_leiden_resolution}_niche_composition{file_suffix}.svg"
    
    df_counts = (model.adata.obs.groupby([latent_cluster_key, cell_type_key])
                 .size().unstack())
    df_counts.plot(kind="bar", stacked=True, figsize=(10,10))
    legend = plt.legend(bbox_to_anchor=(1, 1), loc="upper left", prop={'size': 10})
    legend.set_title("Cell Type Annotations", prop={'size': 10})
    plt.title("Cell Type Composition of Niches")
    plt.xlabel("Niche")
    plt.ylabel("Cell Type Counts")
    if save_fig:
        plt.savefig(file_path,
                    bbox_extra_artists=(legend,),
                    bbox_inches="tight")
    plt.show()


def run_differential_gp_analysis(
    model: CustomNicheCompass,
    latent_cluster_key: str,
    sample_key: str,
    samples: list,
    gp_names_key: str,
    differential_gp_test_results_key: str,
    figure_folder_path: str,
    selected_cats: Optional[list] = None,
    comparison_cats: str = "rest",
    log_bayes_factor_thresh: float = 2.3,
    latent_cluster_colors: Optional[dict] = None,
    save_fig: bool = True,
    save_file: bool = True,
    file_suffix: str = "",
    n_top_enriched_gp_start_idx: int = 0,
    n_top_enriched_gp_end_idx: int = 10,
    n_top_genes_per_gp: int = 3,
    n_top_peaks_per_gp: int = 3
):
    """
    Run differential GP testing and create visualizations.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    latent_cluster_key : str
        Key in adata.obs for niche/cluster annotations
    sample_key : str
        Key in adata.obs for sample annotations
    samples : list
        List of sample names for visualization
    gp_names_key : str
        Key in adata.uns for gene program names
    differential_gp_test_results_key : str
        Key in adata.uns for differential GP test results
    figure_folder_path : str
        Path to save figures
    selected_cats : Optional[list]
        Selected categories for differential testing (None for all)
    comparison_cats : str or list
        Categories to compare against ("rest" or specific list)
    log_bayes_factor_thresh : float
        Threshold for log Bayes factor
    latent_cluster_colors : Optional[dict]
        Color dictionary for clusters (created if None)
    save_fig : bool
        Whether to save figures
    save_file : bool
        Whether to save CSV summary
    file_suffix : str
        Suffix to add to filenames
    n_top_enriched_gp_start_idx : int
        Start index for top enriched GPs to plot
    n_top_enriched_gp_end_idx : int
        End index for top enriched GPs to plot
    n_top_genes_per_gp : int
        Number of top genes per GP to visualize
    n_top_peaks_per_gp : int
        Number of top peaks per GP to visualize
        
    Returns
    -------
    enriched_gps : list
        List of enriched gene program names
    gp_summary_df : pd.DataFrame
        Gene program summary dataframe
    """
    # Check number of active GPs
    active_gps = model.get_active_gps()
    print(f"Number of total gene programs: {len(model.adata.uns[gp_names_key])}.")
    print(f"Number of active gene programs: {len(active_gps)}.")
    
    # Get GP summary
    gp_summary_df = model.get_gp_summary()
    print("\nExample active GPs:")
    display(gp_summary_df[gp_summary_df["gp_active"] == True].head())
    
    # Run differential gp testing
    enriched_gps = model.run_differential_gp_tests(
        cat_key=latent_cluster_key,
        selected_cats=selected_cats,
        comparison_cats=comparison_cats,
        log_bayes_factor_thresh=log_bayes_factor_thresh)
    
    # Results are stored in a df in the adata object
    print("\nDifferential GP test results:")
    display(model.adata.uns[differential_gp_test_results_key])
    
    # Visualize GP activities of enriched GPs across niches
    df = model.adata.obs[[latent_cluster_key] + enriched_gps].groupby(latent_cluster_key).mean()
    
    scaler = MinMaxScaler()
    normalized_columns = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_columns, columns=df.columns)
    normalized_df.index = df.index
    
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(normalized_df,
                     cmap='viridis',
                     annot=False,
                     linewidths=0)
    plt.xticks(rotation=45,
               fontsize=8,
               ha="right")
    plt.xlabel("Gene Programs", fontsize=16)
    plt.ylabel("Niches", fontsize=16)
    heatmap_path = f"{figure_folder_path}/enriched_gps_heatmap{file_suffix}.svg"
    if save_fig:
        plt.savefig(heatmap_path, bbox_inches="tight")
    plt.show()
    
    # Store gene program summary of enriched gene programs
    gp_summary_cols = ["gp_name",
                       "n_source_genes",
                       "n_non_zero_source_genes",
                       "n_target_genes",
                       "n_non_zero_target_genes",
                       "gp_source_genes",
                       "gp_target_genes",
                       "gp_source_genes_importances",
                       "gp_target_genes_importances",
                       "n_source_peaks",
                       "n_target_peaks",
                       "gp_source_peaks",
                       "gp_target_peaks",
                       "gp_source_peaks_importances",
                       "gp_target_peaks_importances"]
    
    enriched_gp_summary_df = gp_summary_df[gp_summary_df["gp_name"].isin(enriched_gps)].copy()
    if len(enriched_gp_summary_df) > 0:
        cat_dtype = pd.CategoricalDtype(categories=enriched_gps, ordered=True)
        enriched_gp_summary_df.loc[:, "gp_name"] = enriched_gp_summary_df["gp_name"].astype(cat_dtype)
        enriched_gp_summary_df = enriched_gp_summary_df.sort_values(by="gp_name")
        enriched_gp_summary_df = enriched_gp_summary_df[gp_summary_cols]
        
        file_path = f"{figure_folder_path}/log_bayes_factor_{log_bayes_factor_thresh}_niche_enriched_gps_summary{file_suffix}.csv"
        if save_file:
            enriched_gp_summary_df.to_csv(file_path)
        else:
            display(enriched_gp_summary_df)
    
    # Generate plots of enriched GPs
    if selected_cats is not None and len(selected_cats) > 0:
        plot_label = f"log_bayes_factor_{log_bayes_factor_thresh}_cluster_{selected_cats[0]}_vs_rest{file_suffix}"
    else:
        plot_label = f"log_bayes_factor_{log_bayes_factor_thresh}_all_clusters{file_suffix}"
    
    if latent_cluster_colors is None:
        latent_cluster_colors = create_new_color_dict(
            adata=model.adata,
            cat_key=latent_cluster_key)
    
    generate_enriched_gp_info_plots(
        plot_label=plot_label,
        model=model,
        sample_key=sample_key,
        differential_gp_test_results_key=differential_gp_test_results_key,
        cat_key=latent_cluster_key,
        cat_palette=latent_cluster_colors,
        n_top_enriched_gp_start_idx=n_top_enriched_gp_start_idx,
        n_top_enriched_gp_end_idx=n_top_enriched_gp_end_idx,
        feature_spaces=samples,
        n_top_genes_per_gp=n_top_genes_per_gp,
        n_top_peaks_per_gp=n_top_peaks_per_gp,
        save_figs=save_fig,
        figure_folder_path=f"{figure_folder_path}/",
        spot_size=30)
    
    return enriched_gps, gp_summary_df


def analyze_cell_cell_communication(
    model: CustomNicheCompass,
    gp_name: str,
    latent_cluster_key: str,
    latent_cluster_colors: dict,
    figure_folder_path: str,
    n_neighbors: int = 4,
    save: bool = True,
    file_suffix: str = ""
):
    """
    Analyze cell-cell communication using a specific gene program.
    
    Parameters
    ----------
    model : CustomNicheCompass
        Trained NicheCompass model
    gp_name : str
        Name of the gene program to analyze
    latent_cluster_key : str
        Key in adata.obs for niche/cluster annotations
    latent_cluster_colors : dict
        Color dictionary for clusters
    figure_folder_path : str
        Path to save figures
    n_neighbors : int
        Number of neighbors for network computation
    save : bool
        Whether to save the figure
    file_suffix : str
        Suffix to add to filename
        
    Returns
    -------
    network_df : pd.DataFrame
        Communication network dataframe
    """
    network_df = compute_communication_gp_network(
        gp_list=[gp_name],
        model=model,
        group_key=latent_cluster_key,
        n_neighbors=n_neighbors)
    
    visualize_communication_gp_network(
        adata=model.adata,
        network_df=network_df,
        figsize=(9, 8),
        cat_colors=latent_cluster_colors,
        edge_type_colors=["#1f77b4"],
        cat_key=latent_cluster_key,
        save=save,
        save_path=f"{figure_folder_path}/gp_network_{gp_name}{file_suffix}.svg",
    )
    
    return network_df


#%% Load trained model

#load_timestamp = "28012026_153505"
load_timestamp = current_timestamp # uncomment if you trained the model in this notebook

model_folder_path = f"{outpath}/artifacts/multimodal/{load_timestamp}/model"
model_folder_stable_path = model_folder_path.replace('artifacts', 'stable')
model_folder_path = model_folder_stable_path if os.path.exists(model_folder_stable_path) else model_folder_path
print(f"Loading model from {model_folder_path}...")

source_model = CustomNicheCompass.load(
    dir_path=model_folder_path,
    adata=None,
    adata_file_name="adata.h5ad",
    adata_atac=None,
    adata_atac_file_name="adata_atac.h5ad",
    gp_names_key=gp_names_key
)

source_samples = source_model.adata.obs[sample_key].unique().tolist()

target_model = CustomNicheCompass.load(
    dir_path=model_folder_path,
    adata=sc.pp.subsample(target_rna, n_obs=10000, copy=True),     # could also use geosketch: sketch_indices = gs(adata.obsm['X_pca'], 5000, replace=False)
    adata_atac=sc.pp.subsample(target_atac, n_obs=10000, copy=True),
    gp_names_key=gp_names_key
)

#target_samples = target_model.adata.obs[target_sample_key].unique().tolist()

#%% Compute neighbor graph and UMAP embedding for target data

sc.pp.neighbors(target_model.adata,
                use_rep=latent_key,
                key_added=latent_key)

sc.tl.umap(target_model.adata,
           neighbors_key=latent_key)

#%% Compute latent representation and neighbor graph & UMAP embedding for SOURCE data

z_source_rna, _, z_source_atac, _, clip_embeddings_rna, clip_embeddings_atac = model.get_latent_representation(
    adata=source_model.adata,
    adata_atac=source_model.adata_atac,
    paired_data=True,
    counts_key="counts",
    adj_key="spatial_connectivities",
    cat_covariates_keys=None,
    only_active_gps=True,
    return_mu_std=True,
    separate_modalities=True,
    return_clip_embeddings=True,
    node_batch_size=source_model.node_batch_size_,
)

clip_embeddings_rna_magnitude = np.linalg.norm(clip_embeddings_rna, axis=1)
clip_embeddings_atac_magnitude = np.linalg.norm(clip_embeddings_atac, axis=1)
print(f'Mean magnitude of clip embeddings - RNA: {np.mean(clip_embeddings_rna_magnitude):.2f}, ATAC: {np.mean(clip_embeddings_atac_magnitude):.2f}')

# Normalize clip embeddings to unit norm
clip_embeddings_rna = clip_embeddings_rna / np.linalg.norm(clip_embeddings_rna, axis=1, keepdims=True)
clip_embeddings_atac = clip_embeddings_atac / np.linalg.norm(clip_embeddings_atac, axis=1, keepdims=True)
assert np.allclose(np.linalg.norm(clip_embeddings_rna, axis=1), 1), "RNA clip embeddings are not unit norm"
assert np.allclose(np.linalg.norm(clip_embeddings_atac, axis=1), 1), "ATAC clip embeddings are not unit norm"


clip_embeddings_adata = ad.AnnData(
    X=np.concatenate([clip_embeddings_rna, clip_embeddings_atac], axis=0),
    obs=pd.concat([
        source_model.adata.obs.assign(modality="rna"),
        source_model.adata_atac.obs.assign(modality="atac"),
    ], axis=0),
    obsm={'X_gps': np.concatenate([z_source_rna, z_source_atac], axis=0)}
)

sc.pp.pca(clip_embeddings_adata, n_comps=50)
sc.pp.neighbors(clip_embeddings_adata, use_rep='X_pca', n_neighbors=100)
sc.tl.leiden(clip_embeddings_adata, resolution=0.5) # also leiden clustering in identify_niches()
sc.tl.umap(clip_embeddings_adata, min_dist=0.3)
sc.pl.umap(clip_embeddings_adata, color=['modality', 'cell_type', 'RNA_clusters', 'ATAC_clusters'], ncols=2, wspace=0.1, size=25)
sc.pl.umap(clip_embeddings_adata, color=['modality', 'leiden', 'RNA_clusters', 'ATAC_clusters'], ncols=2, wspace=0.1, size=25)

sc.tl.embedding_density(clip_embeddings_adata, groupby='modality')
sc.pl.embedding_density(clip_embeddings_adata, key='umap_density_modality')

#%% Compute latent representation and neighbor graph & UMAP embedding for TARGET data
print(f"Computing latent representation for target data (n cells: {target_model.adata.n_obs + target_model.adata_atac.n_obs})...")
'''
compute_and_store_latent_representation(
    model=target_model,
    latent_key=latent_key,
    counts_key=counts_key,
    adj_key=adj_key,
    only_active_gps=True,
    paired_data=False  # Adjust if RNA and ATAC are not paired
)
'''

mu_target_rna, _, mu_target_atac, _, clip_embeddings_rna, clip_embeddings_atac = \
    target_model.get_latent_representation(
                adata=target_model.adata,
                adata_atac=target_model.adata_atac,
                paired_data=False,
                counts_key="counts",
                adj_key="spatial_connectivities",
                cat_covariates_keys=None,
                only_active_gps=True,
                return_mu_std=True,
                separate_modalities=True,
                return_clip_embeddings=True,
                node_batch_size=target_model.node_batch_size_,
        )

clip_embeddings_rna_magnitude = np.linalg.norm(clip_embeddings_rna, axis=1)
clip_embeddings_atac_magnitude = np.linalg.norm(clip_embeddings_atac, axis=1)
print(f'Mean magnitude of raw CLIP embeddings - RNA: {np.mean(clip_embeddings_rna_magnitude):.2f}, ATAC: {np.mean(clip_embeddings_atac_magnitude):.2f}')

# Normalize clip embeddings to unit norm
clip_embeddings_rna = clip_embeddings_rna / np.linalg.norm(clip_embeddings_rna, axis=1, keepdims=True)
clip_embeddings_atac = clip_embeddings_atac / np.linalg.norm(clip_embeddings_atac, axis=1, keepdims=True)
assert np.allclose(np.linalg.norm(clip_embeddings_rna, axis=1), 1), "RNA clip embeddings are not unit norm"
assert np.allclose(np.linalg.norm(clip_embeddings_atac, axis=1), 1), "ATAC clip embeddings are not unit norm"

n_rna = target_model.adata.n_obs

# RNA is always first in the union
clip_rna = clip_embeddings_rna[:n_rna, :]

# ATAC: map atac_obs into the union index
rna_obs = target_model.adata.obs_names
atac_obs = target_model.adata_atac.obs_names
extra_atac_obs = atac_obs[~atac_obs.isin(rna_obs)]
union_obs = rna_obs.append(extra_atac_obs)
atac_pos = union_obs.get_indexer(atac_obs)

clip_atac = clip_embeddings_atac[atac_pos, :]

target_model.adata.obsm[latent_key] = clip_rna
target_model.adata_atac.obsm[latent_key] = clip_atac


clip_embeddings_adata = ad.AnnData(
    X=np.concatenate([clip_rna, clip_atac], axis=0),
    obs=pd.concat([
        target_model.adata.obs.assign(modality="rna"),
        target_model.adata_atac.obs.assign(modality="atac"),
    ], axis=0),
    #obsm={'X_gps': np.concatenate([mu_target_rna, mu_target_atac], axis=0)} # need to correct shape mismatch
)

sc.pp.pca(clip_embeddings_adata, n_comps=50)
sc.pp.neighbors(clip_embeddings_adata, use_rep='X_pca', n_neighbors=100)
sc.tl.leiden(clip_embeddings_adata, resolution=0.5) # also leiden clustering in identify_niches()
sc.tl.umap(clip_embeddings_adata, min_dist=0.3)
sc.pl.umap(clip_embeddings_adata, color=['modality', 'leiden', 'Main_cluster_name'], ncols=3, wspace=0.1, size=25)

sc.tl.embedding_density(clip_embeddings_adata, groupby='modality')
sc.pl.embedding_density(clip_embeddings_adata, key='umap_density_modality')

#%% Save target model

target_model_folder_path = model_folder_path.replace('model', 'target_model')
os.makedirs(target_model_folder_path, exist_ok=True)

target_model.save(
    dir_path=target_model_folder_path,
    overwrite=True,
    save_adata=True,
    adata_file_name="target_adata.h5ad",
    save_adata_atac=True,
    adata_atac_file_name=f"target_adata_atac.h5ad"
)


#%% 4.1 Visualize NicheCompass Latent GP Space (Source)

# Let's look at the preservation of cell type annotations in the latent GP space. 
# Note that the goal of NicheCompass is not a separation of cell types but rather 
# to identify spatially consistent cell niches.

source_cell_type_colors = visualize_cell_types_in_latent_and_physical_space(
    model=source_model,
    samples=source_samples,
    cell_type_key=cell_type_key,
    sample_key=sample_key,
    latent_key=latent_key,
    figure_folder_path=figure_folder_path,
    spot_size=spot_size,
    groups=None,
    save_fig=True,
    file_suffix="_source")

target_cell_type_colors = visualize_cell_types_in_latent_and_physical_space(
    model=target_model,
    samples=target_samples,
    cell_type_key="Main_cluster_name",
    sample_key="PCR_sample_name",
    latent_key=latent_key,
    figure_folder_path=figure_folder_path,
    spot_size=spot_size,
    groups=None,
    save_fig=True,
    file_suffix="_target")


#%% 4.2 Identify Niches (Source)

# We compute Leiden clustering of the NicheCompass latent GP space to identify 
# spatially consistent cell niches.

latent_leiden_resolution = 0.5

source_latent_cluster_colors = identify_niches(
    model=source_model,
    samples=source_samples,
    sample_key=sample_key,
    latent_key=latent_key,
    latent_cluster_key=latent_cluster_key,
    latent_leiden_resolution=latent_leiden_resolution,
    figure_folder_path=figure_folder_path,
    spot_size=spot_size,
    groups=None,
    save_fig=True,
    file_suffix="_source")


#%% 4.3 Characterize niche composition (Source)

characterize_niche_composition(
    model=source_model,
    latent_cluster_key=latent_cluster_key,
    cell_type_key=cell_type_key,
    latent_leiden_resolution=latent_leiden_resolution,
    figure_folder_path=figure_folder_path,
    save_fig=True,
    file_suffix="_source")


#%% 4.3.2 Differential GPs (Source)

# Now we can test which GPs are differentially expressed in a niche. To this end, 
# we will perform differential GP testing of a selected niche, e.g. niche "9" 
# (```selected_cats = ["9"]```) vs all other niches (```comparison_cats = "rest"```). 
# However, differential GP testing can also be performed in the following ways:
# - Set ```selected_cats = None``` to perform differential GP testing across all niches, 
#   as opposed to just for one specific niche.
# - Set ```comparison_cats = ["12"]``` to perform differential GP testing against 
#   niche "12" as opposed to against all other niches.
# 
# We choose an absolute log bayes factor threshold of 2.3 to determine strongly 
# enriched GPs (see https://en.wikipedia.org/wiki/Bayes_factor).

# Set parameters for differential gp testing
selected_cats = ["9"]
comparison_cats = "rest"
log_bayes_factor_thresh = 2.3

source_enriched_gps, source_gp_summary_df = run_differential_gp_analysis(
    model=source_model,
    latent_cluster_key=latent_cluster_key,
    sample_key=sample_key,
    samples=source_samples,
    gp_names_key=gp_names_key,
    differential_gp_test_results_key=differential_gp_test_results_key,
    figure_folder_path=figure_folder_path,
    selected_cats=selected_cats,
    comparison_cats=comparison_cats,
    log_bayes_factor_thresh=log_bayes_factor_thresh,
    latent_cluster_colors=source_latent_cluster_colors,
    save_fig=True,
    save_file=True,
    file_suffix="_source",
    n_top_enriched_gp_start_idx=0,
    n_top_enriched_gp_end_idx=10,
    n_top_genes_per_gp=3,
    n_top_peaks_per_gp=3)


#%% 4.3.3 Cell-cell Communication (Source)

# Now we will use the inferred activity of an enriched combined interaction GP 
# to analyze the involved intercellular interactions.

gp_name = "Cldn11_ligand_receptor_target_gene_GP"

source_network_df = analyze_cell_cell_communication(
    model=source_model,
    gp_name=gp_name,
    latent_cluster_key=latent_cluster_key,
    latent_cluster_colors=source_latent_cluster_colors,
    figure_folder_path=figure_folder_path,
    n_neighbors=n_neighbors,
    save=True,
    file_suffix="_source")


#%% Apply same analyses to target data (optional)

# Uncomment and modify as needed to run analyses on target data:

# target_cell_type_colors = visualize_cell_types_in_latent_and_physical_space(
#     model=target_model,
#     samples=target_samples,
#     cell_type_key=cell_type_key,  # Adjust if different key name
#     sample_key="PCR_sample_name",  # Adjust to match target data
#     latent_key=latent_key,
#     figure_folder_path=figure_folder_path,
#     spot_size=spot_size,
#     groups=None,
#     save_fig=True,
#     file_suffix="_target")
#
# target_latent_cluster_colors = identify_niches(
#     model=target_model,
#     samples=target_samples,
#     sample_key="PCR_sample_name",  # Adjust to match target data
#     latent_key=latent_key,
#     latent_cluster_key=latent_cluster_key,
#     latent_leiden_resolution=latent_leiden_resolution,
#     figure_folder_path=figure_folder_path,
#     spot_size=spot_size,
#     groups=None,
#     save_fig=True,
#     file_suffix="_target")
#
# characterize_niche_composition(
#     model=target_model,
#     latent_cluster_key=latent_cluster_key,
#     cell_type_key=cell_type_key,  # Adjust if different key name
#     latent_leiden_resolution=latent_leiden_resolution,
#     figure_folder_path=figure_folder_path,
#     save_fig=True,
#     file_suffix="_target")
#
# target_enriched_gps, target_gp_summary_df = run_differential_gp_analysis(
#     model=target_model,
#     latent_cluster_key=latent_cluster_key,
#     sample_key="PCR_sample_name",  # Adjust to match target data
#     samples=target_samples,
#     gp_names_key=gp_names_key,
#     differential_gp_test_results_key=differential_gp_test_results_key,
#     figure_folder_path=figure_folder_path,
#     selected_cats=selected_cats,
#     comparison_cats=comparison_cats,
#     log_bayes_factor_thresh=log_bayes_factor_thresh,
#     latent_cluster_colors=target_latent_cluster_colors,
#     save_fig=True,
#     save_file=True,
#     file_suffix="_target",
#     n_top_enriched_gp_start_idx=0,
#     n_top_enriched_gp_end_idx=10,
#     n_top_genes_per_gp=3,
#     n_top_peaks_per_gp=3)
#
# target_network_df = analyze_cell_cell_communication(
#     model=target_model,
#     gp_name=gp_name,
#     latent_cluster_key=latent_cluster_key,
#     latent_cluster_colors=target_latent_cluster_colors,
#     figure_folder_path=figure_folder_path,
#     n_neighbors=n_neighbors,
#     save=True,
#     file_suffix="_target")

#%% End of Notebook

# Cells below are not part of Tutorial

#%%
