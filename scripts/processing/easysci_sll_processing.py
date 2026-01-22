import os
import pandas as pd
import anndata as ad

datapath = "/home/dmannk/links/projects/def-liyue/dmannk/BAKLAVAS_base/data/EasySci_SLL"
mouse_rna_path = os.path.join(datapath, "mouse", "RNA")
mouse_atac_path = os.path.join(datapath, "mouse", "ATAC")

# load RNA-seq data and metadata
rna_gene_count = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_gene_count.txt.gz"), compression="gzip", sep="\t")
rna_exon_count = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_exon_count.txt.gz"), compression="gzip", sep="\t")
rna_cell_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_cell_annotation.csv.gz"), compression="gzip")
rna_exon_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_exon_annotation.csv.gz"), compression="gzip")
rna_gene_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_gene_annotation.csv.gz"), compression="gzip")

# load ATAC-seq data and metadata
atac_gene_activity = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_gene_activity.txt.gz"), compression="gzip", sep="\t")
atac_peak_count = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_peak_count.txt.gz"), compression="gzip", sep="\t")
atac_gene_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_gene_annotation.csv.gz"), compression="gzip")
atac_peak_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_peak_annotation.csv.gz"), compression="gzip")
atac_cell_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_cell_annotation.csv.gz"), compression="gzip")

# create AnnData objects
rna_adata = ad.AnnData(
    X=rna_gene_count.values,
    obs=rna_cell_anno,
    var=rna_gene_anno,
)
atac_adata = ad.AnnData(
    X=atac_gene_activity.values,
    obs=atac_cell_anno,
    var=atac_gene_anno,
)

# save AnnData objects
rna_adata.write_h5ad(os.path.join(mouse_rna_path, "mouse_rna.h5ad"))
atac_adata.write_h5ad(os.path.join(mouse_atac_path, "mouse_atac.h5ad"))
