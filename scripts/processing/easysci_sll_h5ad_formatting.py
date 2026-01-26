import os
import pandas as pd
import anndata as ad
import scipy.io as sio
import numpy as np
import shutil

datapath = "/home/dmannk/links/projects/def-liyue/dmannk/BAKLAVAS_base/data/EasySci_SLL"
mouse_rna_path = os.path.join(datapath, "mouse", "RNA")
mouse_atac_path = os.path.join(datapath, "mouse", "ATAC")

# load RNA-seq data and metadata
#rna_exon_count_adata = sio.mmread(os.path.join(mouse_rna_path, "GSM6538356_RNA_exon_count.txt.gz")).tocsr()
rna_gene_count = sio.mmread(os.path.join(mouse_rna_path, "GSM6538356_RNA_gene_count.txt.gz")).tocsr()
rna_cell_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_cell_annotation.csv.gz"), compression="gzip")
rna_exon_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_exon_annotation.csv.gz"), compression="gzip")
rna_gene_anno = pd.read_csv(os.path.join(mouse_rna_path, "GSM6538356_RNA_gene_annotation.csv.gz"), compression="gzip")

# load ATAC-seq data and metadata
#atac_gene_activity = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_gene_activity.txt.gz"), compression="gzip", sep="\t")
#atac_peak_count = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_peak_count.txt.gz"), compression="gzip", sep="\t")
#atac_gene_activity = sio.mmread(os.path.join(mouse_atac_path, "GSM6538357_ATAC_gene_activity.txt.gz")).tocsr()
#atac_gene_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_gene_annotation.csv.gz"), compression="gzip")
atac_peak_count = sio.mmread(os.path.join(mouse_atac_path, "GSM6538357_ATAC_peak_count.txt.gz")).tocsr()
atac_peak_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_peak_annotation.csv.gz"), compression="gzip")
atac_cell_anno = pd.read_csv(os.path.join(mouse_atac_path, "GSM6538357_ATAC_cell_annotation.csv.gz"), compression="gzip")

# create AnnData objects
rna_adata = ad.AnnData(
    X=rna_gene_count.T,
    obs=rna_cell_anno,
    var=rna_gene_anno,
)
atac_adata = ad.AnnData(
    X=atac_peak_count.T,
    obs=atac_cell_anno,
    var=atac_peak_anno,
)

def atomic_write_h5ad(adata, out_path, name=None):
    """Write AnnData to H5AD atomically via tmp file in SLURM scratch or /tmp, with compression."""
    scratch = os.environ.get("SLURM_TMPDIR", "/tmp")
    base = os.path.basename(out_path) if name is None else name
    tmp_h5ad = os.path.join(scratch, base)
    adata.write_h5ad(tmp_h5ad, compression="gzip", compression_opts=4)
    shutil.move(tmp_h5ad, out_path)
    print("wrote:", out_path)

atomic_write_h5ad(rna_adata, os.path.join(mouse_rna_path, "mouse_rna.h5ad"))
atomic_write_h5ad(atac_adata, os.path.join(mouse_atac_path, "mouse_atac.h5ad"))