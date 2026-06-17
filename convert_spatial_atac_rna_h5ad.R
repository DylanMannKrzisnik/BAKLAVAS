library(SeuratDisk)
library(Seurat)
library(scCustomize)

datapath <- '/Users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/human/'
seurat_obj <- readRDS(paste0(datapath, 'humanbrain_spatial_RNA_ATAC.rds'))

## inspect data
Assays(seurat_obj)
Images(seurat_obj)
Reductions(seurat_obj)

## export each assay to h5ad
assays <- Assays(seurat_obj)

for (a in assays) {
  message("Exporting assay: ", a)
  
  # Create a minimal Seurat object with only this assay
  tmp <- subset(seurat_obj, assay = a)
  DefaultAssay(tmp) <- a
  
  # Optional: drop reductions to avoid conversion warnings
  tmp@reductions <- list()
  
  # File name
  h5s_file <- paste0(datapath, a, ".h5Seurat")
  
  # Save + convert
  SaveH5Seurat(tmp, filename = h5s_file, overwrite = TRUE)
  Convert(h5s_file, dest = "h5ad", overwrite = TRUE)
}

# Save image
img <- seurat_obj@images$slice1

png::writePNG(img@image, paste0(datapath, "tissue_hires_image.png"))
coords <- GetTissueCoordinates(seurat_obj, image = "slice1")
write.csv(coords, paste0(datapath, "spatial_coords.csv"), row.names = TRUE)

img@scale.factors
saveRDS(img@scale.factors, paste0(datapath, "scale_factors.rds"))

