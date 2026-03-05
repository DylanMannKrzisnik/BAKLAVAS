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

#%% Load environment variables
from dotenv import load_dotenv, dotenv_values
load_dotenv()

from pprint import pprint
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values())

#%% 1.1 Import Libraries

import os
import warnings
from datetime import datetime
from typing import Optional

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
import mlflow
import anndata as ad

from nichecompass.utils import (create_new_color_dict,
                                compute_communication_gp_network,
                                visualize_communication_gp_network,
                                generate_enriched_gp_info_plots)

#%% Import custom NicheCompass and DataAligner classes

import sys

# Allow importing `nichecompass_utils.py` from repo root when running this script
sys.path.append(os.environ.get('BAKLAVA_ROOT'))
from nichecompass_utils import CustomNicheCompass

#%% 1.2 Define Parameters


### Dataset ###
dataset = "Spatial_ATAC_RNA"
datapath = os.path.join(os.environ.get('DATAPATH'), dataset)
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
conv_layer_encoder = "gcnconv" # default is "gatv2conv", change to "gcnconv" if not enough compute and memory

# Trainer
n_epochs = 5
n_epochs_all_gps = 5
lr = 0.001
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_chrom_access_recon = 300.
lambda_l1_masked = 0. # prior GP  regularization
lambda_l1_addon = 30. # de novo GP regularization
use_cuda_if_available = True

### Analysis ###
cell_type_key = "cell_type"
latent_leiden_resolution = 0.6
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 30
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

#%% Configure Paths

# Define paths
outpath = "/home/mcb/users/dmannk/BAKLAVA_base/outputs/nichecompass_mouse_brain_multimodal"
os.makedirs(outpath, exist_ok=True)

# Get time of notebook execution for timestamping saved artifacts
now = datetime.now()
current_timestamp = now.strftime("%d%m%Y_%H%M%S")

ga_data_folder_path = f"{os.environ.get('DATAPATH')}/gene_annotations"
gp_data_folder_path = f"{os.environ.get('DATAPATH')}/gene_programs"
#so_data_folder_path = f"{datapath}/spatial_omics"
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
os.makedirs(gp_data_folder_path, exist_ok=True)
os.makedirs(ga_data_folder_path, exist_ok=True)
#os.makedirs(so_data_folder_path, exist_ok=True)


#%% 1.6 Download Files (Optional)
# You can skip this part if you have downloaded the files mentioned above manually, or you are using your own data.
'''
if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain_atac.h5ad')):
    gdown.download("https://drive.google.com/file/d/1WH_9PYV_AEcLd5QVNILig-_gfS-EdRVR', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain_atac.h5ad")
if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain.h5ad')):
    gdown.download("https://drive.google.com/file/d/1NpRynEDnGnxab6sHJy4AeitSKnwfX2Mi', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain.h5ad")
if not os.path.exists(os.path.join(so_data_folder_path, 'spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv')):
    gdown.download("https://drive.google.com/file/d/1w3eorJojnndhmeTicbin7eRjR8Pwj7O-', so_data_folder_path+'/spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv")
'''

#%% 2. MODEL PREPARATION

# - NicheCompass expects a prior GP mask as input, which it will use to make its latent feature space interpretable (through linear masked decoders). 
# - The user can provide a custom GP mask to NicheCompass based on the biological question of interest.
# - As a default, we create a GP mask based on four databases of prior knowledge of inter- and intracellular interaction pathways:
#     - OmniPath (Ligand-Receptor GPs)
#     - MEBOCOST (Enzyme-Sensor GPs)
#     - CollecTRI (Transcriptional Regulation GPs)
#     - NicheNet (Combined Interaction GPs)

#%% Load / prepare input data (cached or freshly processed)

# Toggle between using a cached prepared dataset (fast) vs rebuilding from raw
# source/target files (slow, but reproducible).
USE_CACHED_DATA = True

# Optional override for cache directory (timestamp folder). If None, uses latest.
CACHE_DIR_OVERRIDE = None

from scripts.hpo_mouse_brain_utils import (
    resolve_hparams,
    filter_model_hparams,
    filter_train_hparams,
)

if USE_CACHED_DATA:
    from scripts.hpo_mouse_brain_utils import (
        resolve_cache_dir,
        load_cached_inputs,
        load_cached_targets,
    )

    cache_dir = resolve_cache_dir(CACHE_DIR_OVERRIDE)
    adata, adata_atac = load_cached_inputs(cache_dir)
    target_rna, target_atac = load_cached_targets(cache_dir)

    source_name = "cached_data"
    target_name = "cached_data"

else:
    # Fresh rebuild from raw files.
    from data_utils import (
        load_spatial_atac_rna_mouse_brain_source,
        load_mousedev_spatial_triomic_data,
        load_10x_mouse_brain_ad_data,
        load_10x_mouse_brain_data,
        basic_feature_processing_for_alignment,
        annotate_genes_and_peaks_for_alignment,
        filter_spatially_variable_features,
        align_source_target_multimodal_by_overlap,
        build_combined_gp_dict_mouse_brain,
        rebuild_nichecompass_multimodal_masks,
        finalize_target_after_alignment,
        add_pseudocount_layers,
    )
    
    adata, adata_atac, source_assembly, source_name = load_spatial_atac_rna_mouse_brain_source(
        so_data_folder_path = f"/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics",
        dataset = "spatial_atac_rna_seq_mouse_brain",
        cell_type_key = cell_type_key,
        spatial_key = spatial_key,
        n_neighbors = n_neighbors,
        adj_key = adj_key,
    )
    target_rna, target_atac, target_assembly, target_name = load_10x_mouse_brain_data()

    # Build prior GPs (needed for filtering - tutorial section 2.1)
    combined_gp_dict = build_combined_gp_dict_mouse_brain(
        species=species,
        omnipath_lr_network_file_path=omnipath_lr_network_file_path,
        nichenet_lr_network_file_path=nichenet_lr_network_file_path,
        nichenet_ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        mebocost_enzyme_sensor_interactions_folder_path=mebocost_enzyme_sensor_interactions_folder_path,
        collectri_tf_network_file_path=collectri_tf_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        verbose=True,
    )

    # Filter source data to spatially variable genes/peaks (tutorial section 2.3)
    adata, adata_atac = filter_spatially_variable_features(
        adata=adata,
        adata_atac=adata_atac,
        combined_gp_dict=combined_gp_dict,
        filter_genes=filter_genes,
        filter_peaks=filter_peaks,
        n_svg=n_svg,
        n_svp=n_svp,
        min_cell_gene_thresh_ratio=min_cell_gene_thresh_ratio,
        min_cell_peak_thresh_ratio=min_cell_peak_thresh_ratio,
        adj_key=adj_key,
        verbose=True,
    )

    # Add genomic coordinates required for peak-overlap alignment (tutorial section 2.4)
    adata, adata_atac = annotate_genes_and_peaks_for_alignment(
        adata=adata, adata_atac=adata_atac, gtf_file_path=gtf_file_path
    )

    # Match the previously inlined alignment preprocessing
    target_rna.var_names = target_rna.var_names.str.split(".").str[0]
    target_rna = target_rna[:, ~target_rna.var_names.duplicated(keep="first")]

    if "peak" in target_atac.var.columns and not set(["chrom", "chromStart", "chromEnd"]).issubset(target_atac.var.columns):
        target_atac.var[["chrom", "chromStart", "chromEnd"]] = target_atac.var["peak"].str.split(":|-").tolist()

    adata, adata_atac, target_rna, target_atac = basic_feature_processing_for_alignment(
        adata=adata,
        adata_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
    )


    adata, adata_atac, target_rna, target_atac = align_source_target_multimodal_by_overlap(
        adata=adata,
        adata_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        source_name=source_name,
        target_name=target_name,
        source_assembly=source_assembly,
        target_assembly=target_assembly,
    )

    # Ensure target data exposes `spatial_connectivities` expected by NicheCompass
    target_rna, target_atac = finalize_target_after_alignment(
        adata=adata,
        adata_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        adj_type="knn",
    )

    # Rebuild all required NicheCompass masks (GP + chromatin accessibility) on aligned feature space
    adata, adata_atac, target_rna, target_atac = rebuild_nichecompass_multimodal_masks(
        adata=adata,
        adata_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        combined_gp_dict=combined_gp_dict,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        adj_key=adj_key,
        filter_peaks_based_on_genes=True,
    )

    # Ensure ATAC modality has the spatial connectivities expected by the model.
    if adj_key in adata.obsp:
        adata_atac.obsp[adj_key] = adata.obsp[adj_key].copy()

    add_pseudocount_layers(adata, adata_atac, target_rna, target_atac)


#%% Initialize model

hparam_defaults = resolve_hparams()
model_hparams = filter_model_hparams(hparam_defaults)

model = CustomNicheCompass(
    adata,
    adata_atac,
    **model_hparams,
)


#%% Train model

# Keep MLflow in ONE place so "mlflow ui" consistently shows all runs.
# Default: directory above the repo (i.e., BAKLAVA_base), but can be overridden
# via MLFLOW_BASE_DIR env var (same convention as scripts/hpo_optuna_mouse_brain.py).
mlflow_base_dir = os.environ.get("MLFLOW_BASE_DIR")
mlflow_base_dir = os.path.abspath(mlflow_base_dir)

mlflow_db_path = os.path.join(mlflow_base_dir, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
print(f"MLflow backend-store-uri: sqlite:////{mlflow_db_path.lstrip('/')}")

# Set up MLflow experiment for logging
mlflow_experiment_name = f"nichecompass_mouse_brain_multimodal"
mlflow_artifact_dir = os.path.join(mlflow_base_dir, "mlflow_artifacts", mlflow_experiment_name)
os.makedirs(mlflow_artifact_dir, exist_ok=True)

# Get or create experiment
mlflow_experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
if mlflow_experiment is None:
    # Create new experiment
    mlflow_experiment_id = mlflow.create_experiment(
        mlflow_experiment_name,
        artifact_location=os.path.abspath(mlflow_artifact_dir),
    )
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

    ## Get train kwargs from hparam defaults
    train_kwargs = filter_train_hparams(hparam_defaults)

    ## Update with user inputs
    train_kwargs.update({
        "n_epochs": n_epochs,
        "n_epochs_all_gps": n_epochs_all_gps,
        "lr": lr,
        "lambda_edge_recon": lambda_edge_recon,
        "lambda_gene_expr_recon": lambda_gene_expr_recon,
        "lambda_chrom_access_recon": lambda_chrom_access_recon,
        "lambda_l1_masked": lambda_l1_masked,
        "lambda_l1_addon": lambda_l1_addon,
        "edge_batch_size": 64,
        "node_batch_size": 500,
        "use_cuda_if_available": use_cuda_if_available,
        "n_sampled_neighbors": n_sampled_neighbors,
        "target_adata": target_rna,
        "target_adata_atac": target_atac,
        "target_holdout_frac": 0.1,
        "target_holdout_n": 500,
        "target_holdout_seed": 0,
        "target_paired_data": True,
        "target_encoder_input_key": train_kwargs.get("encoder_input_key"),
        "target_counts_key": counts_key,
        "log_target_multimodal_contrastive": True,
        "use_early_stopping": False,
        "verbose": False,
        "mlflow_experiment_id": mlflow_experiment_id,
    })
    

    ## Train model
    model.train(**train_kwargs)


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

#source_samples = source_model.adata.obs[sample_key].unique().tolist()

target_model = CustomNicheCompass.load(
    dir_path=model_folder_path,
    adata=sc.pp.subsample(target_rna, n_obs=10000, copy=True),     # could also use geosketch: sketch_indices = gs(adata.obsm['X_pca'], 5000, replace=False)
    adata_atac=sc.pp.subsample(target_atac, n_obs=10000, copy=True),
    gp_names_key=gp_names_key
)

#target_samples = target_model.adata.obs[target_sample_key].unique().tolist()

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
    only_active_gps=False,
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
                only_active_gps=False,
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
#sc.pl.umap(clip_embeddings_adata, color=['modality', 'leiden', 'Main_cluster_name'], ncols=3, wspace=0.1, size=25)
sc.pl.umap(clip_embeddings_adata, color=['modality', 'leiden'], ncols=3, wspace=0.1, size=25)

sc.tl.embedding_density(clip_embeddings_adata, groupby='modality')
sc.pl.embedding_density(clip_embeddings_adata, key='umap_density_modality')


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
