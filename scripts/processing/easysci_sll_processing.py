# %% load libraries
import os
import subprocess
import tempfile
import shutil
import anndata as ad
import scanpy as sc
from muon import atac as ac
import mudata
import mygene

# %% load data
datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL"
mouse_rna_path = os.path.join(datapath, "mouse", "RNA", "mouse_rna.h5ad")
mouse_atac_path = os.path.join(datapath, "mouse", "ATAC", "mouse_atac.h5ad")

mouse_rna = ad.read_h5ad(mouse_rna_path)
mouse_atac = ad.read_h5ad(mouse_atac_path)

if "counts" not in mouse_rna.layers:
    mouse_rna.layers["counts"] = mouse_rna.X.copy()
if "counts" not in mouse_atac.layers:
    mouse_atac.layers["counts"] = mouse_atac.X.copy()

mdata = mudata.MuData({"rna": mouse_rna, "atac": mouse_atac})

print(mouse_rna.shape)
print(mouse_atac.shape)

# %% add genomic intervals for genes (needed for Muon ATAC gene annotation)
#
# Notes from the interactive session:
# - `mouse_rna.var['gene_id']` contains Ensembl IDs with version (e.g. ENSMUSG...1)
# - Muon expects an `interval` column like "chr8:222-333" for genes
# - MyGene's `genomic_pos` is assembly-dependent; if you need mm10 specifically,
#   switch to `genomic_pos_mm10` instead.

mg = mygene.MyGeneInfo()

if "gene_id" in mouse_rna.var.columns:
    gene_ids = mouse_rna.var["gene_id"].astype(str)
elif "gene_ids" in mouse_rna.var.columns:
    gene_ids = mouse_rna.var["gene_ids"].astype(str)
else:
    raise KeyError("Expected `mouse_rna.var['gene_id']` or `mouse_rna.var['gene_ids']`.")

mouse_rna.var["gene_id_no_version"] = gene_ids.str.split(".").str[0]
genes = mouse_rna.var["gene_id_no_version"].tolist()

genome_field = "genomic_pos"  # mm39 by default; use "genomic_pos_mm10" if needed
results = mg.querymany(
    genes,
    scopes="ensembl.gene",
    species="mouse",
    fields=genome_field,
    as_dataframe=True,
)

results.dropna(
    subset=[f"{genome_field}.chr", f"{genome_field}.start", f"{genome_field}.end"],
    inplace=True,
)
intervals = (
    results.apply(
        lambda x: (
            f"chr{x[f'{genome_field}.chr']}:{int(x[f'{genome_field}.start'])}-{int(x[f'{genome_field}.end'])}"
        ),
        axis=1,
    )
    .rename("interval")
    .drop_duplicates()
)

mouse_rna.var = mouse_rna.var.merge(
    intervals.to_frame(),
    left_on="gene_id_no_version",
    right_index=True,
    how="left",
)
mouse_rna.var["interval"] = mouse_rna.var["interval"].fillna("chrNaN:0-1")

if "gene_id" in mouse_rna.var.columns and "gene_ids" not in mouse_rna.var.columns:
    # Keep original `gene_id` but also provide the `gene_ids` alias used by some tooling.
    mouse_rna.var["gene_ids"] = mouse_rna.var["gene_id"]

# %% process RNA data
mouse_rna.var['mt'] = mouse_rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(mouse_rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pp.calculate_qc_metrics(mouse_rna, percent_top=None, log1p=False, inplace=True)
sc.pl.violin(mouse_rna, ['total_counts', 'n_genes_by_counts'], jitter=0.4, multi_panel=True)

sc.pp.filter_cells(mouse_rna, min_genes=100)
sc.pp.filter_genes(mouse_rna, min_cells=3)

sc.pp.normalize_total(mouse_rna, target_sum=1e4)
sc.pp.log1p(mouse_rna)
sc.pp.highly_variable_genes(mouse_rna, n_top_genes=2000)

sc.tl.pca(mouse_rna, svd_solver="arpack", use_highly_variable=True)
sc.pp.neighbors(mouse_rna, use_rep="X_pca")

#sc.tl.umap(mouse_rna)
#sc.pl.umap(mouse_rna, color=["Main_cluster_name", "Replicate_ID"], wspace=0.2)


# %% process ATAC data
sc.pp.calculate_qc_metrics(mouse_atac, percent_top=None, log1p=False, inplace=True)
sc.pl.violin(mouse_atac, ['total_counts', 'n_genes_by_counts'], jitter=0.4, multi_panel=True)

sc.pp.filter_cells(mouse_atac, min_genes=100)
sc.pp.filter_genes(mouse_atac, min_cells=10)

# Derive gene annotations and compute TSS enrichment (from the interactive session).
_orig_var_names = mouse_rna.var_names.copy()
try:
    if "gene_short_name" in mouse_rna.var.columns:
        # Muon is happiest when genes are indexed by gene symbol/name.
        mouse_rna.var_names = mouse_rna.var["gene_short_name"].astype(str).values
        mouse_rna.var_names_make_unique()
    features = ac.tl.get_gene_annotation_from_rna(mouse_rna)
finally:
    mouse_rna.var_names = _orig_var_names

#tss = ac.tl.tss_enrichment(mouse_atac, n_tss=1000, features=features)
# ac.pl.tss_enrichment(tss)

# Optional: HV peak selection/plotting on raw counts (kept from the interactive session)
sc.pp.normalize_total(mouse_atac, target_sum=1e4)
sc.pp.log1p(mouse_atac)
sc.pp.highly_variable_genes(mouse_atac, min_mean=0.05, max_mean=1.5, min_disp=0.5)
sc.pl.highly_variable_genes(mouse_atac)

# LSI pipeline (from the interactive session)
ac.pp.tfidf(mouse_atac, scale_factor=1e4)
ac.tl.lsi(mouse_atac)

# Drop the first LSI component (often correlated with sequencing depth)
mouse_atac.obsm["X_lsi"] = mouse_atac.obsm["X_lsi"][:, 1:]
mouse_atac.varm["LSI"] = mouse_atac.varm["LSI"][:, 1:]
mouse_atac.uns["lsi"]["stdev"] = mouse_atac.uns["lsi"]["stdev"][1:]

sc.pp.neighbors(mouse_atac, use_rep="X_lsi")

# %% save outputs
compression_opts = 4    # the higher the value, the more compression: default is 4, max is 9
mouse_rna.write_h5ad(os.path.join(datapath, "mouse", "RNA", "mouse_rna_processed.h5ad"), compression="gzip", compression_opts=compression_opts)
mouse_atac.write_h5ad(os.path.join(datapath, "mouse", "ATAC", "mouse_atac_processed.h5ad"), compression="gzip", compression_opts=compression_opts)
#mdata = mudata.MuData({"rna": mouse_rna, "atac": mouse_atac})
#mdata.write_h5mu(os.path.join(datapath, "mouse", "mouse_mudata_processed.h5mu"), compression="gzip", compression_opts=compression_opts)
