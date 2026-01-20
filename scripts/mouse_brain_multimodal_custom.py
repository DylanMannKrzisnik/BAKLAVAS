"""
Minimal multimodal setup for CustomNicheCompass.

Loads paired RNA/ATAC AnnData objects, ensures a spatial neighbor graph exists,
and instantiates a CustomNicheCompass model ready for training.
"""

from pathlib import Path

import scanpy as sc
import squidpy as sq

from nichecompass_utils import CustomNicheCompass


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "Spatial_ATAC_RNA" / "mouse"

    rna_path = data_root / "spatial_atac_rna_seq_mouse_brain.h5ad"
    atac_path = data_root / "spatial_atac_rna_seq_mouse_brain_atac.h5ad"

    adata = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # Ensure the spatial neighbor graph exists.
    adj_key = "spatial_connectivities"
    if adj_key not in adata.obsp:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            spatial_key="spatial",
            n_neighs=4,
        )
        adata.obsp[adj_key] = adata.obsp[adj_key].maximum(
            adata.obsp[adj_key].T)

    # Ensure counts layer exists for RNA.
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Sanity checks for required GP mask keys.
    required_varm = [
        "nichecompass_gp_targets",
        "nichecompass_gp_sources",
        "nichecompass_gp_targets_categories",
        "nichecompass_gp_sources_categories",
    ]
    missing = [key for key in required_varm if key not in adata.varm]
    if missing:
        raise ValueError(
            "Missing GP mask keys in adata.varm: "
            f"{missing}. Add GP masks before training."
        )

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
