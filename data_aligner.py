#%%
import anndata as ad
from muon import MuData
from pybedtools import BedTool
import numpy as np
from nichecompass_utils import CustomNicheCompass
import pandas as pd

'''
import mygene as mg

model_folder_path = "/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/artifacts/multimodal/21012026_181240/model"
gp_names_key = "nichecompass_gp_names"

model = CustomNicheCompass.load(
    dir_path=model_folder_path,
    adata=None,
    adata_file_name="adata.h5ad",
    adata_atac=None,
    adata_atac_file_name="adata_atac.h5ad",
    gp_names_key=gp_names_key
)

source_rna = model.adata
source_atac = model.adata_atac
source_data = MuData({"rna": source_rna, "atac": source_atac})

target_rna = ad.read_h5ad("/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL/mouse/RNA/mouse_rna_processed.h5ad")
target_atac = ad.read_h5ad("/home/mcb/users/dmannk/BAKLAVA_base/data/EasySci_SLL/mouse/ATAC/mouse_atac_processed.h5ad")

mginfo = mg.MyGeneInfo()
results = mginfo.querymany(
    target_rna.var["gene_id_no_version"].tolist(),
    scopes="ensembl.gene",
    species="mouse",
    fields="symbol",
    as_dataframe=True,
)
results = results.reset_index().drop_duplicates(subset='query') # remove duplicate genes
assert results.groupby('query')['symbol'].nunique().le(1).all(), "Multiple symbols still found for some genes"

target_rna.var = target_rna.var.merge(results, left_on="gene_id_no_version", right_on="query", how="left")
target_rna.var.loc[target_rna.var['symbol'].isna(), 'symbol'] = target_rna.var.loc[target_rna.var['symbol'].isna(), 'query']
target_rna.var.set_index("symbol", inplace=True)

## remove duplicate genes (again)
target_rna = target_rna[:, ~target_rna.var_names.duplicated(keep='first')]
assert target_rna.var_names.is_unique, "Target RNA data must have unique gene names"

target_atac.var[['chrom', 'chromStart', 'chromEnd']] = target_atac.var['peak'].str.split('-').tolist()

target_data = MuData({"rna": target_rna, "atac": target_atac})

'''

class DataAligner:
    def __init__(
        self,
        source_data: ad.AnnData | MuData,
        target_data: ad.AnnData | MuData
    ):

        self.source_data = source_data
        self.target_data = target_data

        ## check that all gene names are unique
        assert self.source_data['rna'].var_names.is_unique, "Source RNA data must have unique gene names"
        assert self.target_data['rna'].var_names.is_unique, "Target RNA data must have unique gene names"

        ## check that all gene names are str and not castable as int
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.source_data['rna'].var_names), "Gene names must be strings and not castable as int"
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.target_data['rna'].var_names), "Gene names must be strings and not castable as int"

        ## check proper column names for ATAC peak intervals
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.source_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.target_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"

        ## set var_names to BED format for ATAC data
        self.source_data['atac'].var_names = self.source_data['atac'].var.apply(lambda x: f"{x['chrom']}:{x['chromStart']}-{x['chromEnd']}", axis=1)
        self.target_data['atac'].var_names = self.target_data['atac'].var.apply(lambda x: f"{x['chrom']}:{x['chromStart']}-{x['chromEnd']}", axis=1)

    def find_gene_overlap(self):
        source_genes = self.source_data['rna'].var_names.tolist()
        target_genes = self.target_data['rna'].var_names.tolist()
        self.gene_overlap = list(set(source_genes) & set(target_genes))

    def find_peak_overlap(self, peak_name_mapper_as_dict: bool = False):
        source_atac = self.source_data['atac']
        target_atac = self.target_data['atac']
        source_peaks = BedTool.from_dataframe(source_atac.var[["chrom", "chromStart", "chromEnd"]])
        target_peaks = BedTool.from_dataframe(target_atac.var[["chrom", "chromStart", "chromEnd"]])
        peak_overlap = source_peaks.intersect(target_peaks, wa=True, wb=True).to_dataframe() # Intersect with -wa -wb to return intervals from both sets

        # Group by target peak coordinates and select the row with highest "dispersion_norm" within each group
        # First, rename columns to identify source and target peaks
        peak_overlap.rename(columns={
            "chrom": "chrom_source", "start": "start_source", "end": "end_source",
            "name": "chrom_target", "score": "start_target", "strand": "end_target"
        }, inplace=True)
        
        idx_target = None
        idx_source = None

        if "dispersions_norm" in target_atac.var.columns:
            merged_target = (
                peak_overlap[['chrom_target', 'start_target', 'end_target']].astype(str)
                .merge(
                    target_atac.var.reset_index().astype(str),
                    left_on=['chrom_target', 'start_target', 'end_target'],
                    right_on=['chrom', 'chromStart', 'chromEnd'],
                    how='left'
                )
            )
            idx_target = merged_target.groupby(['chrom', 'chromStart', 'chromEnd'])['dispersions_norm'].idxmax()

        if "dispersions_norm" in source_atac.var.columns:
            merged_source = (
                peak_overlap[['chrom_source', 'start_source', 'end_source']].astype(str)
                .merge(
                    source_atac.var.reset_index().astype(str),
                    left_on=['chrom_source', 'start_source', 'end_source'],
                    right_on=['chrom', 'chromStart', 'chromEnd'],
                    how='left'
                )
            )
            idx_source = merged_source.groupby(['chrom', 'chromStart', 'chromEnd'])['dispersions_norm'].idxmax()

        if (idx_target is not None) and (idx_source is not None):
            idx = pd.concat([idx_target, idx_source])
        elif idx_target is not None:
            idx = idx_target
        elif idx_source is not None:
            idx = idx_source
        else:
            idx = None

        peak_overlap = peak_overlap.loc[idx.values]

        peak_name_mapper = peak_overlap.assign(
            peak_name_source = peak_overlap.apply(lambda x: f"{x['chrom_source']}:{x['start_source']}-{x['end_source']}", axis=1),
            peak_name_target = peak_overlap.apply(lambda x: f"{x['chrom_target']}:{x['start_target']}-{x['end_target']}", axis=1)
        ).set_index('peak_name_source').loc[:,'peak_name_target']

        print(f"Number of duplicate peak name mappings: {peak_name_mapper.index.value_counts().ge(2).sum()}")
        self.peak_overlap_df = peak_overlap
        self.peak_name_mapper = peak_name_mapper.to_dict() if peak_name_mapper_as_dict else peak_name_mapper # to_dict() ignored because of duplicate values


    def align_features_by_overlap(self):
        source_rna = self.source_data['rna']
        target_rna = self.target_data['rna']
        source_atac = self.source_data['atac']
        target_atac = self.target_data['atac']

        source_rna = source_rna[:, source_rna.var_names.get_indexer(self.gene_overlap)]
        target_rna = target_rna[:, target_rna.var_names.get_indexer(self.gene_overlap)]
        assert (source_rna.var_names == target_rna.var_names).all(), "Source and target RNA data must have the same gene names"

        target_atac = target_atac[:, target_atac.var_names.get_indexer(self.peak_name_mapper.values)]
        source_atac = source_atac[:, source_atac.var_names.get_indexer(self.peak_name_mapper.index)]

        source_atac.var_names = source_atac.var_names.map(self.peak_name_mapper.to_dict())
        print('Proportion of overlapping peak names: ', (source_atac.var_names == target_atac.var_names).mean())
        #assert (source_atac.var_names == target_atac.var_names).all(), "Source and target ATAC data must have the same peak names"

        self.source_data = MuData({"rna": source_rna, "atac": source_atac})
        self.target_data = MuData({"rna": target_rna, "atac": target_atac})


#%%
data_aligner = DataAligner(source_data, target_data)
data_aligner.find_gene_overlap()
data_aligner.find_peak_overlap()
data_aligner.align_features_by_overlap()
