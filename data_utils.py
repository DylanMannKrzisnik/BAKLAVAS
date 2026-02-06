from __future__ import annotations

import os
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = [
    "multimodal_latents_adata",
    "load_spatial_atac_rna_mouse_brain_source",
    "load_10x_mouse_brain_ad_data",
    "basic_feature_processing_for_alignment",
    "annotate_genes_and_peaks_for_alignment",
    "align_source_target_multimodal_by_overlap",
    "finalize_target_after_alignment",
    "add_pseudocount_layers",
    "fix_var_for_h5ad",
]


def multimodal_latents_adata(modality1_dict, modality2_dict, latent_key):
    """Build a single AnnData from two modality-specific latent spaces."""

    adata1 = list(modality1_dict.values())[0]
    adata2 = list(modality2_dict.values())[0]
    modality1_name = list(modality1_dict.keys())[0]
    modality2_name = list(modality2_dict.keys())[0]

    obs = pd.concat(
        [
            adata1.obs.assign(modality=modality1_name),
            adata2.obs.assign(modality=modality2_name),
        ],
        axis=0,
    )

    latents = np.concatenate([adata1.obsm[latent_key], adata2.obsm[latent_key]], axis=0)

    return sc.AnnData(X=latents, obs=obs)


def load_spatial_atac_rna_mouse_brain_source(
    so_data_folder_path: str,
    dataset: str,
    *,
    cell_type_annotations_csv: Optional[str] = None,
    cell_type_key: str = "cell_type",
    cell_type_col_in_csv: str = "predicted.celltype",
    atac_clusters_col_in_csv: str = "ATAC_clusters",
    spatial_key: str = "spatial",
    n_neighbors: int = 4,
    adj_key: str = "spatial_connectivities",
    make_adjacency_symmetric: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load the paired source RNA/ATAC spatial mouse brain dataset.

    Returns
    -------
    (adata_rna, adata_atac, source_assembly, source_name)
    """

    # Local import: squidpy is optional in some environments.
    import squidpy as sq

    adata = sc.read_h5ad(os.path.join(so_data_folder_path, f"{dataset}.h5ad"))
    adata_atac = sc.read_h5ad(os.path.join(so_data_folder_path, f"{dataset}_atac.h5ad"))

    if cell_type_annotations_csv is None:
        cell_type_annotations_csv = os.path.join(
            so_data_folder_path, f"{dataset}_cell_type_annotations.csv"
        )

    if os.path.isfile(cell_type_annotations_csv):
        cell_type_df = pd.read_csv(cell_type_annotations_csv, index_col=0)
        if cell_type_col_in_csv in cell_type_df.columns:
            cell_type_df = cell_type_df.rename({cell_type_col_in_csv: cell_type_key}, axis=1)
        if atac_clusters_col_in_csv in cell_type_df.columns:
            cell_type_df = cell_type_df.drop(atac_clusters_col_in_csv, axis=1)
        adata.obs = adata.obs.merge(cell_type_df, left_index=True, right_index=True, how="left")

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        spatial_key=spatial_key,
        n_neighs=n_neighbors,
    )

    if make_adjacency_symmetric and (adj_key in adata.obsp):
        adata.obsp[adj_key] = adata.obsp[adj_key].maximum(adata.obsp[adj_key].T)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if "counts" not in adata_atac.layers:
        adata_atac.layers["counts"] = adata_atac.X.copy()

    return adata, adata_atac, "mm10", "Spatial_ATAC_RNA"


def load_10x_mouse_brain_ad_data(
    data_dir: Optional[str] = None,
    base_name: str = "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote",
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load 10x Multiome (RNA+ATAC) mouse brain AD dataset.

    Expected files in `data_dir`:
    - {base_name}_filtered_feature_bc_matrix.h5  (required)
    - {base_name}_atac_peaks.bed                (optional)
    - {base_name}_atac_peak_annotation.tsv      (optional)

    Returns
    -------
    (target_rna, target_atac, target_assembly, target_name)
    """

    if data_dir is None:
        # data/ folder is one level above the repo folder (BAKLAVA/)
        repo_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(repo_dir, "..", "data", "10x_mouse_brain_AD")
    data_dir = os.path.abspath(data_dir)

    h5_path = os.path.join(data_dir, f"{base_name}_filtered_feature_bc_matrix.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"10x filtered feature-barcode matrix not found: {h5_path}. "
            "Download it from 10x into data_dir."
        )

    adata_full = sc.read_10x_h5(h5_path, gex_only=False)
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

    for a in (adata_rna, adata_atac):
        if ft_col in a.var.columns:
            a.var = a.var.drop(columns=[ft_col])

    adata_rna.layers["counts"] = adata_rna.X.copy()
    adata_atac.layers["counts"] = adata_atac.X.copy()

    peaks_bed_path = os.path.join(data_dir, f"{base_name}_atac_peaks.bed")
    peak_ann_path = os.path.join(data_dir, f"{base_name}_atac_peak_annotation.tsv")

    if os.path.isfile(peaks_bed_path):
        bed = pd.read_csv(
            peaks_bed_path,
            sep="	",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "chromStart", "chromEnd"],
        )
        peak_id_bed = bed["chrom"].astype(str) + "-" + bed["chromStart"].astype(str) + "-" + bed["chromEnd"].astype(str)

        if adata_atac.n_vars == len(peak_id_bed) and (adata_atac.var_names == peak_id_bed.values).all():
            adata_atac.var["chrom"] = bed["chrom"].values
            adata_atac.var["chromStart"] = bed["chromStart"].values
            adata_atac.var["chromEnd"] = bed["chromEnd"].values
        else:
            adata_atac.var["peak"] = adata_atac.var_names
            coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
            adata_atac.var["chrom"] = coords[0]
            adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
            adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""
    else:
        adata_atac.var["peak"] = adata_atac.var_names
        coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
        adata_atac.var["chrom"] = coords[0]
        adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
        adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""

    if os.path.isfile(peak_ann_path):
        ann = pd.read_csv(peak_ann_path, sep="	")
        peak_col = "Peak" if "Peak" in ann.columns else ("peak" if "peak" in ann.columns else None)
        if peak_col is not None:
            ann_indexed = ann.set_index(peak_col)
            extra = [c for c in ann_indexed.columns if c not in adata_atac.var.columns]
            if extra:
                adata_atac.var = adata_atac.var.merge(
                    ann_indexed[extra], left_index=True, right_index=True, how="left"
                )

    return adata_rna, adata_atac, "mm10", "10x_mouse_brain_AD"


def basic_feature_processing_for_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    target_min_genes: int = 100,
    target_min_cells: int = 3,
    n_pcs: int = 50,
    target_knn: int = 100,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Basic preprocessing used before overlap-based alignment."""

    for obj, name in [
        (adata, "source_rna"),
        (adata_atac, "source_atac"),
        (target_rna, "target_rna"),
        (target_atac, "target_atac"),
    ]:
        if "counts" not in obj.layers:
            raise KeyError(f"Counts layer not found in {name}.layers['counts']")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)

    sc.pp.filter_cells(target_rna, min_genes=target_min_genes)
    sc.pp.filter_genes(target_rna, min_cells=target_min_cells)
    sc.pp.normalize_total(target_rna, target_sum=1e4)
    sc.pp.log1p(target_rna)

    sc.pp.filter_cells(target_atac, min_genes=target_min_genes)
    sc.pp.filter_genes(target_atac, min_cells=target_min_cells)
    sc.pp.normalize_total(target_atac, target_sum=1e4)
    sc.pp.log1p(target_atac)

    sc.pp.pca(target_rna, n_comps=n_pcs)
    sc.pp.neighbors(target_rna, use_rep="X_pca", n_neighbors=target_knn)

    sc.pp.pca(target_atac, n_comps=n_pcs)
    sc.pp.neighbors(target_atac, use_rep="X_pca", n_neighbors=target_knn)

    return adata, adata_atac, target_rna, target_atac


def align_source_target_multimodal_by_overlap(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    source_name: str,
    target_name: str,
    source_assembly: str,
    target_assembly: str,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Align source and target by gene and peak overlap using `DataAligner`."""

    from muon import MuData
    from data_aligner import DataAligner

    source_data = MuData({"rna": adata, "atac": adata_atac})
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

    return (
        data_aligner.source_data["rna"],
        data_aligner.source_data["atac"],
        data_aligner.target_data["rna"],
        data_aligner.target_data["atac"],
    )


def finalize_target_after_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    adj_type: str = "knn",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Copy annotations from source to target and set `spatial_connectivities`."""

    from data_aligner import DataAligner

    target_rna, target_atac = DataAligner.copy_annotations_to_target(
        source_rna=adata,
        source_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
    )
    target_rna, target_atac = DataAligner.set_target_spatial_connectivities(
        target_rna=target_rna,
        target_atac=target_atac,
        adj_type=adj_type,
    )
    return target_rna, target_atac


def add_pseudocount_layers(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    pseudocount_key: str = "pseudocounts",
) -> None:
    """Create shallow references to `.X` in `.layers[pseudocount_key]` (in-place)."""

    adata.layers[pseudocount_key] = adata.X
    adata_atac.layers[pseudocount_key] = adata_atac.X
    target_rna.layers[pseudocount_key] = target_rna.X
    target_atac.layers[pseudocount_key] = target_atac.X


def fix_var_for_h5ad(adata_obj: ad.AnnData) -> ad.AnnData:
    """Convert non-string columns in `.var` to string representation (best-effort)."""

    for col in list(adata_obj.var.columns):
        try:
            if adata_obj.var[col].dtype == "object":

                def safe_str_convert(x):
                    if pd.isna(x) or x is None:
                        return ""
                    if isinstance(x, str):
                        return x
                    try:
                        return str(x)
                    except Exception:
                        return ""

                adata_obj.var[col] = adata_obj.var[col].apply(safe_str_convert).astype(str)
            elif hasattr(adata_obj.var[col].dtype, "categories"):
                adata_obj.var[col] = adata_obj.var[col].astype(str)
        except Exception:
            adata_obj.var = adata_obj.var.drop(columns=[col])

    return adata_obj


def annotate_genes_and_peaks_for_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    *,
    gtf_file_path: str,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Add gene and peak genomic coordinates using a GTF file."""

    from nichecompass.utils import get_gene_annotations

    return get_gene_annotations(adata=adata, adata_atac=adata_atac, gtf_file_path=gtf_file_path)
