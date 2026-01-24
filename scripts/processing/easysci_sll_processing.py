#%%
import os
import anndata as ad
import scanpy as sc
from muon import atac as ac
import mudata


datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL"
mouse_rna_path = os.path.join(datapath, "mouse", "RNA", "mouse_rna.h5ad")
mouse_atac_path = os.path.join(datapath, "mouse", "ATAC", "mouse_atac.h5ad")

#%% load data
mouse_rna = ad.read_h5ad(mouse_rna_path)
mouse_atac = ad.read_h5ad(mouse_atac_path)

mouse_rna.layers["counts"] = mouse_rna.X.copy()
mouse_atac.layers["counts"] = mouse_atac.X.copy()

mdata = mudata.MuData({"rna": mouse_rna, "atac": mouse_atac})

print(mouse_rna.shape)
print(mouse_atac.shape)

#%% get genomic position of a gene
import mygene

mg = mygene.MyGeneInfo()

# Your list of genes
genes = mouse_rna.var['gene_id'].str.split('.').str[0].tolist()

# 1. Bulk query for mm10 positions
# We specifically request 'genomic_pos_mm10' to avoid the newer mm39
results = mg.querymany(genes, 
                       scopes='symbol,ensembl.gene', 
                       species='mouse',
                       fields='genomic_pos',
                       as_dataframe=True)

results.dropna(subset=['genomic_pos.chr', 'genomic_pos.start', 'genomic_pos.end'], inplace=True)
intervals = results.apply(lambda x: f"chr{x['genomic_pos.chr']}:{int(x['genomic_pos.start'])}-{int(x['genomic_pos.end'])}", axis=1)
print("WARNING: mouse genome assembly not specified")

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

#mouse_atac.obs['NS']=1
#ac.pl.fragment_histogram(mouse_atac, region='chr1:1-2000000')
#ac.tl.nucleosome_signal(mouse_atac, n=1e6)

#features = ac.tl.get_gene_annotation_from_rna(mdata)
#tss = ac.tl.tss_enrichment(mdata, n_tss=1000)  # by default, features=ac.tl.get_gene_annotation_from_rna(mdata)
#ac.pl.tss_enrichment(tss)

ac.pp.tfidf(mouse_atac, scale_factor=1e4)

sc.pp.normalize_per_cell(mouse_atac, counts_per_cell_after=1e4)
sc.pp.log1p(mouse_atac)

sc.pp.highly_variable_genes(mouse_atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
sc.pl.highly_variable_genes(mouse_atac)
np.sum(atac.var.highly_variable)

