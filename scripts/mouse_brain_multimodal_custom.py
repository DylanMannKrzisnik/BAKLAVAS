"""
Minimal multimodal setup for CustomNicheCompass (dual-encoder).

This is intentionally lightweight and assumes you've already prepared the
required GP masks in `adata.varm[...]` and (for multimodal) the CA masks in
`adata_atac.varm[...]`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import squidpy as sq

# Allow importing `nichecompass_utils.py` from repo root when running this script
BAKLAVA_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(BAKLAVA_ROOT))

from nichecompass_utils import CustomNicheCompass  # noqa: E402


def main() -> None:
    dataset = "spatial_atac_rna_seq_mouse_brain"
    spatial_key = "spatial"
    n_neighbors = 4

    counts_key = "counts"
    adj_key = "spatial_connectivities"

    gp_names_key = "nichecompass_gp_names"
    active_gp_names_key = "nichecompass_active_gp_names"
    gp_targets_mask_key = "nichecompass_gp_targets"
    gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
    gp_sources_mask_key = "nichecompass_gp_sources"
    gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
    latent_key = "nichecompass_latent"

    data_root = Path(
        os.path.abspath(
            os.path.join(
                str(BAKLAVA_ROOT),
                "..",
                "data",
                "Spatial_ATAC_RNA",
                "mouse",
                "spatial_omics",
            )
        )
    )
    rna_path = data_root / f"{dataset}.h5ad"
    atac_path = data_root / f"{dataset}_atac.h5ad"
    cell_type_path = data_root / f"{dataset}_cell_type_annotations.csv"

    adata = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # Optional: attach cell type annotations if present
    if cell_type_path.exists():
        cell_type_key = "cell_type"
        cell_type_df = pd.read_csv(cell_type_path, index_col=0)
        if "predicted.celltype" in cell_type_df.columns:
            cell_type_df.rename({"predicted.celltype": cell_type_key},
                                axis=1,
                                inplace=True)
        if "ATAC_clusters" in cell_type_df.columns:
            cell_type_df.drop("ATAC_clusters", axis=1, inplace=True)
        adata.obs = adata.obs.merge(
            cell_type_df, left_index=True, right_index=True, how="left"
        )

    # Ensure the spatial neighbor graph exists.
    if adj_key not in adata.obsp:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            spatial_key=spatial_key,
            n_neighs=n_neighbors,
        )
    adata.obsp[adj_key] = adata.obsp[adj_key].maximum(adata.obsp[adj_key].T)

    # Ensure counts layer exists for RNA.
    if counts_key not in adata.layers:
        adata.layers[counts_key] = adata.X.copy()

    # Minimal sanity checks for required GP mask keys.
    required_varm = [
        gp_targets_mask_key,
        gp_sources_mask_key,
        gp_targets_categories_mask_key,
        gp_sources_categories_mask_key,
    ]
    missing = [key for key in required_varm if key not in adata.varm]
    if missing:
        raise ValueError(
            "Missing GP mask keys in adata.varm: "
            f"{missing}. Add GP masks before training."
        )

    model = CustomNicheCompass(
        adata=adata,
        adata_atac=adata_atac,
        counts_key=counts_key,
        adj_key=adj_key,
        gp_names_key=gp_names_key,
        active_gp_names_key=active_gp_names_key,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        latent_key=latent_key,
        conv_layer_encoder="gatv2conv",
        active_gp_thresh_ratio=0.01,
    )

    print("CustomNicheCompass initialized. Ready to train.")
    # Example:
    # model.train(
    #     n_epochs=400,
    #     n_epochs_all_gps=25,
    #     lr=1e-3,
    #     edge_batch_size=256,
    #     n_sampled_neighbors=4,
    #     verbose=True,
    # )


if __name__ == "__main__":
    main()
"""
Minimal multimodal setup for CustomNicheCompass.

Loads paired RNA/ATAC AnnData objects, ensures a spatial neighbor graph exists,
and instantiates a CustomNicheCompass model ready for training.
"""

from pathlib import Path

import scanpy as sc
import squidpy as sq

import sys, os
BAKLAVA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BAKLAVA_ROOT)
from nichecompass_utils import CustomNicheCompass


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
edge_batch_size = 256 # increase if more memory available or decrease to save memory
use_cuda_if_available = True

### Analysis ###
cell_type_key = "cell_type"
latent_leiden_resolution = 0.6
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 30
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

# Define paths
datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse"

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
artifacts_folder_path = f"{datapath}/artifacts"
model_folder_path = f"{artifacts_folder_path}/multimodal/{current_timestamp}/model"
figure_folder_path = f"{artifacts_folder_path}/multimodal/{current_timestamp}/figures"



def main() -> None:
    data_root = os.path.abspath(os.path.join(BAKLAVA_ROOT, "..", "data", "Spatial_ATAC_RNA", "mouse", "spatial_omics"))
    data_root = Path(data_root)
    rna_path = data_root / "spatial_atac_rna_seq_mouse_brain.h5ad"
    atac_path = data_root / "spatial_atac_rna_seq_mouse_brain_atac.h5ad"

    adata = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

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


    #%% 2.5 Add GP Mask to Data

    # Add the GP dictionary as binary masks to the adata
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
        plot_gp_gene_count_distributions=True)


    #%% 2.6 Add Chromatin Accessibility Mask to Data

    # Based on spatial proximity to the genes in the GP mask, we will add a chromatin accessibility mask.

    gene_peak_mapping_dict = generate_multimodal_mapping_dict(
        adata=adata,
        adata_atac=adata_atac)

    adata, adata_atac = add_multimodal_mask_to_adata(
        adata=adata,
        adata_atac=adata_atac,
        gene_peak_mapping_dict=gene_peak_mapping_dict)

    print(f"Keeping {adata_atac.n_vars} peaks after filtering peaks with "
        "no matching genes in gp mask.")

    # Instantiate the model (ready to call model.train(...)).
    model = CustomNicheCompass(
        adata=adata,
        adata_atac=adata_atac,
        counts_key="counts",
        adj_key=adj_key,
        gp_names_key="nichecompass_gp_names",
        active_gp_names_key="nichecompass_active_gp_names",
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        latent_key="nichecompass_latent",
        conv_layer_encoder="gatv2conv",
        active_gp_thresh_ratio=0.01,
    )

    print("CustomNicheCompass initialized. Ready to train.")
    # Example:
    # model.train(n_epochs=400, n_epochs_all_gps=25, lr=1e-3)


if __name__ == "__main__":
    main()
