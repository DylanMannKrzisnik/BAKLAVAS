#!/usr/bin/env Rscript
# Export aligned se.multi.list to Matrix Market + metadata for Python/MuData.
#
# Usage:
#   Rscript export_se_multi_to_mtx.R
#   /Users/dmannk/cisformer/envs/torch_env_py39/bin/python build_mudata_from_mtx.py

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

working.dir <- "/Users/dmannk/BAKLAVA_base/outputs/SMA"
r_dir <- file.path(working.dir, "R_objects")
out_dir <- file.path(working.dir, "h5mu_export")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

se.multi.list <- readRDS(file.path(r_dir, "se.multi.list"))
se.rna.list <- readRDS(file.path(r_dir, "se.RNA.list"))
knn_path <- file.path(r_dir, "knn_spatial_df_filtered_list")
knn_spatial_df_filtered_list <- if (file.exists(knn_path)) readRDS(knn_path) else NULL

if (is.null(names(se.multi.list)) || any(names(se.multi.list) == "")) {
  if (length(se.multi.list) != length(se.rna.list)) {
    stop("se.multi.list and se.RNA.list have different lengths; cannot assign sample names.")
  }
  names(se.multi.list) <- names(se.rna.list)
}

if (!is.null(knn_spatial_df_filtered_list)) {
  if (is.null(names(knn_spatial_df_filtered_list)) || any(names(knn_spatial_df_filtered_list) == "")) {
    if (length(knn_spatial_df_filtered_list) != length(se.multi.list)) {
      stop("knn_spatial_df_filtered_list and se.multi.list have different lengths; cannot assign sample names.")
    }
    names(knn_spatial_df_filtered_list) <- names(se.multi.list)
  }

  missing_samples <- setdiff(names(se.multi.list), names(knn_spatial_df_filtered_list))
  if (length(missing_samples) > 0) {
    stop("Missing alignment metadata for samples: ", paste(missing_samples, collapse = ", "))
  }
}

add_alignment_metadata <- function(obs, alignment, sample_id) {
  if (is.null(alignment)) {
    warning("No alignment metadata found for sample ", sample_id)
    return(obs)
  }

  alignment <- as.data.frame(alignment)
  required_cols <- c("from", "to", "distance", "x", "x_end", "y", "y_end")
  missing_cols <- setdiff(required_cols, colnames(alignment))
  if (length(missing_cols) > 0) {
    stop(
      "Alignment metadata for sample ", sample_id,
      " is missing columns: ", paste(missing_cols, collapse = ", ")
    )
  }

  alignment <- alignment[, required_cols, drop = FALSE]
  colnames(alignment) <- c(
    "msi_barcode", "barcode", "alignment_distance",
    "msi_warped_x", "rna_warped_x", "msi_warped_y", "rna_warped_y"
  )

  if (anyDuplicated(alignment$barcode)) {
    stop("Alignment metadata for sample ", sample_id, " contains duplicate RNA barcodes.")
  }

  obs <- merge(obs, alignment, by = "barcode", all.x = TRUE, sort = FALSE)
  missing_alignment <- sum(is.na(obs$msi_barcode))
  if (missing_alignment > 0) {
    warning(
      "Sample ", sample_id, " has ", missing_alignment,
      " observations without alignment metadata."
    )
  }

  obs
}

export_one <- function(se, sample_id, out_dir, alignment = NULL) {
  sample_dir <- file.path(out_dir, sample_id)
  dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)

  rna <- LayerData(se, assay = "RNA", layer = "counts")
  msi <- LayerData(se, assay = "MSI", layer = "counts")

  if (!all(colnames(rna) == colnames(msi))) {
    stop("RNA and MSI barcodes differ for sample ", sample_id)
  }

  obs <- se@meta.data
  obs$barcode <- rownames(obs)

  if ("Staffli" %in% names(se@tools)) {
    staffli <- se@tools$Staffli@meta.data
    staffli$barcode <- rownames(staffli)
    obs <- merge(obs, staffli, by = "barcode", all.x = TRUE, suffixes = c("", "_staffli"))
  }

  obs$sample_id <- sample_id
  obs <- add_alignment_metadata(obs, alignment, sample_id)

  # AnnData/MuData use cells x features.
  writeMM(t(rna), file.path(sample_dir, "rna.mtx"))
  writeMM(t(msi), file.path(sample_dir, "msi.mtx"))
  write.table(rownames(rna), file.path(sample_dir, "rna_features.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(rownames(msi), file.path(sample_dir, "msi_features.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(colnames(rna), file.path(sample_dir, "barcodes.tsv"),
              quote = FALSE, col.names = FALSE, row.names = FALSE)
  write.table(obs, file.path(sample_dir, "obs.tsv"),
              quote = FALSE, sep = "\t", row.names = FALSE)

  message("Wrote ", sample_dir, " (", ncol(se), " paired spots)")
}

for (sample_id in names(se.multi.list)) {
  alignment <- if (!is.null(knn_spatial_df_filtered_list)) {
    knn_spatial_df_filtered_list[[sample_id]]
  } else {
    NULL
  }
  export_one(se.multi.list[[sample_id]], sample_id, out_dir, alignment)
}

message("Done. Run: /Users/dmannk/cisformer/envs/torch_env_py39/bin/python build_mudata_from_mtx.py")
