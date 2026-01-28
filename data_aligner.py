#%%
import anndata as ad
from muon import MuData
from pybedtools import BedTool
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from pyliftover import LiftOver
from pyranges import PyRanges, read_gtf
import os
import subprocess
from nichecompass_utils import CustomNicheCompass

class DataAligner:
    def __init__(
        self,
        source_data: ad.AnnData | MuData,
        target_data: ad.AnnData | MuData,
        source_assembly: str,
        target_assembly: str,
        source_name: str = "source",
        target_name: str = "target",
    ):

        ## set metadata
        source_data.obs["dataset_name"] = source_name
        target_data.obs["dataset_name"] = target_name
        source_data.obs["assembly"] = source_assembly
        target_data.obs["assembly"] = target_assembly

        ## set TSS if 'rna:strand' is present
        if 'strand' in source_data['rna'].var.columns:
            source_data['rna'].var['TSS'] = source_data['rna'].var.apply(lambda x: x['chromStart'] if x['strand'] == '+' else x['chromEnd'] if x['strand'] == '-' else np.nan, axis=1)
        if 'strand' in target_data['rna'].var.columns:
            target_data['rna'].var['TSS'] = target_data['rna'].var.apply(lambda x: x['chromStart'] if x['strand'] == '+' else x['chromEnd'] if x['strand'] == '-' else np.nan, axis=1)

        ## assign data to self
        self.source_data = source_data
        self.target_data = target_data

        ## check that data are from the same assembly
        if self.source_data.obs["assembly"].equals(self.target_data.obs["assembly"]):
            print(f"Source and target data are from the same assembly: {self.source_data.obs['assembly'].unique()[0]}")
        else:
            print(f"Source and target data are from different assemblies: {self.source_data.obs['assembly'].unique()[0]} and {self.target_data.obs['assembly'].unique()[0]}")
            self.do_liftOver()

        ## check that all gene names are unique
        assert self.source_data['rna'].var_names.is_unique, "Source RNA data must have unique gene names"
        assert self.target_data['rna'].var_names.is_unique, "Target RNA data must have unique gene names"

        ## check that all gene names are str and not castable as int
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.source_data['rna'].var_names), "Gene names must be strings and not castable as int"
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.target_data['rna'].var_names), "Gene names must be strings and not castable as int"

        ## check proper column names for ATAC peak intervals
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.source_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.target_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"

        # Normalize ATAC interval dtypes to avoid string/float formatting mismatches
        for mdata in (self.source_data, self.target_data):
            atac_var = mdata["atac"].var
            atac_var["chrom"] = atac_var["chrom"].astype(str)
            atac_var["chromStart"] = pd.to_numeric(atac_var["chromStart"], errors="raise").astype(int)
            atac_var["chromEnd"] = pd.to_numeric(atac_var["chromEnd"], errors="raise").astype(int)

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

        if idx is not None:
            idx = idx.dropna().astype(int)
            peak_overlap = peak_overlap.loc[idx.values]

        ## could also use mmn.atac.pp.add_positions(mdata['atac'])
        peak_overlap = peak_overlap.assign(
            peak_name_source = peak_overlap.apply(lambda x: f"{x['chrom_source']}:{x['start_source']}-{x['end_source']}", axis=1),
            peak_name_target = peak_overlap.apply(lambda x: f"{x['chrom_target']}:{x['start_target']}-{x['end_target']}", axis=1)
        )

        # Enforce a one-to-one mapping (greedy) to avoid duplicates causing silent drops later
        peak_overlap["overlap_len"] = (
            np.minimum(peak_overlap["end_source"].astype(int), peak_overlap["end_target"].astype(int))
            - np.maximum(peak_overlap["start_source"].astype(int), peak_overlap["start_target"].astype(int))
        )
        peak_overlap = peak_overlap.sort_values("overlap_len", ascending=False)
        peak_overlap = peak_overlap.drop_duplicates(subset=["peak_name_source"], keep="first")
        peak_overlap = peak_overlap.drop_duplicates(subset=["peak_name_target"], keep="first")

        peak_name_mapper = peak_overlap.set_index("peak_name_source").loc[:, "peak_name_target"]

        print(f"Number of duplicate peak name mappings: {peak_name_mapper.index.value_counts().ge(2).sum()}")
        self.peak_overlap_df = peak_overlap
        self.peak_name_mapper = peak_name_mapper.to_dict() if peak_name_mapper_as_dict else peak_name_mapper # to_dict() ignored because of duplicate values


    def align_features_by_overlap(self):
        source_rna = self.source_data['rna']
        target_rna = self.target_data['rna']
        source_atac = self.source_data['atac']
        target_atac = self.target_data['atac']

        # IMPORTANT: avoid get_indexer() here; -1 values silently select the last column and misalign features
        gene_overlap = [g for g in self.gene_overlap if (g in source_rna.var_names) and (g in target_rna.var_names)]
        source_rna = source_rna[:, gene_overlap].copy()
        target_rna = target_rna[:, gene_overlap].copy()
        assert (source_rna.var_names == target_rna.var_names).all(), "Source and target RNA data must have the same gene names"

        peak_name_mapper = self.peak_name_mapper
        if isinstance(peak_name_mapper, dict):
            peak_name_mapper = pd.Series(peak_name_mapper)

        # Filter to mappings that exist in both objects
        peak_name_mapper = peak_name_mapper.loc[
            peak_name_mapper.index.isin(source_atac.var_names)
            & peak_name_mapper.isin(target_atac.var_names)
        ]
        # Ensure uniqueness on both sides (defensive; find_peak_overlap tries to enforce this)
        peak_name_mapper = peak_name_mapper[~peak_name_mapper.index.duplicated(keep="first")]
        peak_name_mapper = peak_name_mapper[~peak_name_mapper.duplicated(keep="first")]

        # Subset by names (not integer positions) so we cannot accidentally select "-1" columns
        source_atac = source_atac[:, peak_name_mapper.index.tolist()].copy()
        target_atac = target_atac[:, peak_name_mapper.values.tolist()].copy()

        # Rename source peaks to the chosen target peak names (so names must match exactly)
        source_atac.var_names = pd.Index(peak_name_mapper.values.tolist())

        print('Proportion of overlapping peak names: ', (source_atac.var_names == target_atac.var_names).mean())
        assert (source_atac.var_names == target_atac.var_names).all(), "Source and target ATAC data must have the same peak names"

        # Preserve obs columns (e.g., 'assembly', 'dataset_name') when creating new MuData objects
        # MuData.obs is derived from modalities, so we need to explicitly preserve columns
        source_obs = self.source_data.obs.copy()  # Use RNA obs as reference (should match ATAC obs)
        target_obs = self.target_data.obs.copy()
        
        # Verify that RNA and ATAC obs indices match (they should for multimodal data)
        assert source_rna.obs_names.equals(source_atac.obs_names), "Source RNA and ATAC must have matching obs_names"
        
        self.source_data = MuData({"rna": source_rna, "atac": source_atac})
        self.target_data = MuData({"rna": target_rna, "atac": target_atac})
        
        # Explicitly set obs to preserve all columns (indices should match)
        assert self.source_data.obs_names.equals(source_obs.index), "MuData obs_names must match preserved obs index"
        assert self.target_data.obs_names.equals(target_obs.index), "MuData obs_names must match preserved obs index"
        self.source_data.obs = source_obs
        self.target_data.obs = target_obs

    def do_liftOver(self):
        print(f"Lifting over source data to target assembly: {self.source_data.obs['assembly'].unique()[0]} to {self.target_data.obs['assembly'].unique()[0]}")

        lo = LiftOver(self.source_data.obs['assembly'].unique()[0], self.target_data.obs['assembly'].unique()[0])
        
        ## lift over whole intervals from source to target assembly
        # NOTE: pyliftover maps single coordinates. For BED-like half-open intervals [start, end),
        # we map start and (end-1), then set new_end = mapped_end + 1.

        # Fetch source ATAC modality
        if hasattr(self.source_data, "mod") and ("atac" in self.source_data.mod):
            source_atac = self.source_data.mod["atac"]
        else:
            try:
                source_atac = self.source_data["atac"]
            except Exception as e:
                raise ValueError("Source data must contain an 'atac' modality to liftOver.") from e

        required_cols = {"chrom", "chromStart", "chromEnd"}
        missing = required_cols - set(source_atac.var.columns)
        if missing:
            raise ValueError(f"Source ATAC .var is missing required columns: {sorted(missing)}")

        var = source_atac.var.copy()
        var["chrom"] = var["chrom"].astype(str)
        var["chromStart"] = pd.to_numeric(var["chromStart"], errors="raise").astype(int)
        var["chromEnd"] = pd.to_numeric(var["chromEnd"], errors="raise").astype(int)

        def _normalize_chr(chrom: str) -> str:
            c = str(chrom)
            if c in {"M", "MT"}:
                c = "chrM"
            if not c.startswith("chr"):
                c = f"chr{c}"
            if c == "chrMT":
                c = "chrM"
            return c

        def _lift_interval(chrom: str, start: int, end: int):
            if end <= start:
                return None
            c = _normalize_chr(chrom)

            start_hits = lo.convert_coordinate(c, int(start))
            end_hits = lo.convert_coordinate(c, int(end) - 1)
            if not start_hits or not end_hits:
                return None

            # Prefer hits on the same target chrom and strand
            best = None
            for (c1, p1, s1, _sc1) in start_hits:
                for (c2, p2, s2, _sc2) in end_hits:
                    if (c1 == c2) and (s1 == s2):
                        best = (c1, int(p1), int(p2), s1)
                        break
                if best is not None:
                    break
            if best is None:
                return None

            new_chrom, p_start, p_end_last, _strand = best
            new_start = min(p_start, p_end_last)
            new_end = max(p_start, p_end_last) + 1  # convert back to half-open
            return new_chrom, new_start, new_end

        def _liftOver_diagnostics(lifted_kept: pd.DataFrame, n_total: int, do_tss_diagnostics: bool = False):

            ## number of mapped peaks
            n_kept = int(lifted_kept.shape[0])
            print(f"[Dx] liftOver mapped {n_kept}/{n_total} source peaks ({(n_kept / max(n_total, 1)):.2%}). Dropping {n_total - n_kept}.")

            ## width correlation
            orig_width = lifted_kept.index.str.split(':').str[1].str.split('-').to_series().apply(
                lambda x: int(x[1]) - int(x[0])).values
            lifted_width = (lifted_kept["chromEnd"] - lifted_kept["chromStart"]).values
            corr, _ = pearsonr(orig_width, lifted_width)
            print(f"[Dx] Correlation between original and lifted width: {corr:.2f}")

            ## TSS distance
            if do_tss_diagnostics:
                print("[PROGRESS] Performing TSS diagnostics...")
                # IMPORTANT:
                # - Parse original peaks from the *index* (original coordinates) and normalize chr naming
                # - Use lifted coordinates from columns
                # - Ensure Start/End are ints (PyRanges expects integer coordinates)
                source_assembly = self.source_data.obs["assembly"].unique()[0]
                target_assembly = self.target_data.obs["assembly"].unique()[0]
                source_tss_pr = _get_reference_tss(source_assembly)
                target_tss_pr = _get_reference_tss(target_assembly)

                orig_df = lifted_kept.index.to_series().str.extract(
                    r"^(?P<Chromosome>[^:]+):(?P<Start>\d+)-(?P<End>\d+)$"
                )
                orig_df["Chromosome"] = orig_df["Chromosome"].map(_normalize_chr)
                orig_df[["Start", "End"]] = orig_df[["Start", "End"]].astype(int)
                # Use stable original peak name as peak_id (robust to reindexing/sorting)
                peak_id = lifted_kept.index.astype(str)
                orig_df["peak_id"] = peak_id.values

                lifted_df = lifted_kept.rename(
                    columns={"chrom": "Chromosome", "chromStart": "Start", "chromEnd": "End"}
                ).copy()
                lifted_df["Chromosome"] = lifted_df["Chromosome"].astype(str).map(_normalize_chr)
                lifted_df[["Start", "End"]] = lifted_df[["Start", "End"]].astype(int)
                lifted_df["peak_id"] = peak_id.values

                # Nearest TSS in each assembly; some peaks on contigs without TSS annotations can be dropped,
                # so we merge on peak_id to compare only peaks present in both results.
                src_near = PyRanges(orig_df).nearest(source_tss_pr).df
                tgt_near = PyRanges(lifted_df).nearest(target_tss_pr).df

                # Keep minimal columns for comparison (gene_name comes from the TSS annotation)
                src_near = src_near.loc[:, ["peak_id", "Distance", "gene_name"]].rename(
                    columns={"Distance": "Distance_source", "gene_name": "gene_name_source"}
                )
                tgt_near = tgt_near.loc[:, ["peak_id", "Distance", "gene_name"]].rename(
                    columns={"Distance": "Distance_target", "gene_name": "gene_name_target"}
                )

                merged = src_near.merge(tgt_near, on="peak_id", how="inner")
                print(f"[Dx] TSS diagnostic comparable peaks: {merged.shape[0]}/{lifted_kept.shape[0]} ({(merged.shape[0]/max(lifted_kept.shape[0],1)):.2%})")

                prop_same_gene = (merged["gene_name_source"].values == merged["gene_name_target"].values).mean()
                print(f"[Dx] Proportion of same nearest gene name: {prop_same_gene:.2%}")

                delta = np.abs(merged["Distance_source"].to_numpy() - merged["Distance_target"].to_numpy())
                med = np.median(delta)
                p95 = np.percentile(delta, 95)

                from scipy.stats import wasserstein_distance
                wd = wasserstein_distance(merged["Distance_source"].to_numpy(), merged["Distance_target"].to_numpy())

                print(f"[Dx] |Δ| median={med:.1f}bp, P95={p95:.1f}bp; Wasserstein={wd:.1f}bp")


        def _get_reference_tss(assembly: str):
            annot_path = "/home/mcb/users/dmannk/BAKLAVA_base/data/reference_tss"
            os.makedirs(annot_path, exist_ok=True)
            parquet_path = os.path.join(annot_path, f"{assembly}_tss.parquet")

            # Check if parquet file already exists
            if os.path.exists(parquet_path):
                info = pd.read_parquet(parquet_path)
            else:
                # Download and process GTF only if parquet doesn't exist
                if assembly == "mm10":
                    gtf_path = os.path.join(annot_path, "mm10_gencode.gtf.gz")
                    if not os.path.exists(gtf_path):
                        annot_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
                        subprocess.run(["wget", annot_url, "-O", gtf_path], check=True)
                    info = read_gtf(gtf_path)

                elif assembly == "mm39":
                    gtf_path = os.path.join(annot_path, "mm39_gencode.gtf.gz")
                    if not os.path.exists(gtf_path):
                        annot_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz"
                        subprocess.run(["wget", annot_url, "-O", gtf_path], check=True)
                    info = read_gtf(gtf_path)

                else:
                    raise ValueError(f"Assembly {assembly} not supported")

                # Process GTF to extract TSS
                info = info[info.df['Feature'] == 'gene'].df
                info["tss"] = info["Start"].where(info["Strand"] == "+", info["End"])
                
                # Save to parquet for future use
                info.to_parquet(parquet_path)

            # Convert to PyRanges format
            tss_pr = PyRanges(
                info[['gene_name', 'Chromosome', 'tss']].assign(
                    tssp1 = info['tss']+1
                ).rename(columns={"tss": "Start", "tssp1": "End"})
            )
            return tss_pr

        ## perform liftOver
        lifted = var.apply(
            lambda r: _lift_interval(r["chrom"], r["chromStart"], r["chromEnd"]),
            axis=1,
        )
        keep_mask = lifted.notna().to_numpy()

        # Subset ATAC to successfully lifted peaks
        kept_index = var.index[keep_mask]
        source_atac = source_atac[:, keep_mask].copy()
        lifted_kept = pd.DataFrame(
            lifted[keep_mask].tolist(),
            index=kept_index,
            columns=["chrom", "chromStart", "chromEnd"],
        )
        # Defensive: ensure we didn't lose alignment when subsetting
        assert source_atac.var.index.equals(kept_index), "liftOver: source_atac.var.index no longer matches kept_index"

        ## perform liftOver diagnostics
        _liftOver_diagnostics(lifted_kept, int(var.shape[0]), do_tss_diagnostics=True)

        # Keep originals for auditing
        source_atac.var["chrom_original"] = source_atac.var["chrom"].astype(str)
        source_atac.var["chromStart_original"] = pd.to_numeric(source_atac.var["chromStart"], errors="coerce")
        source_atac.var["chromEnd_original"] = pd.to_numeric(source_atac.var["chromEnd"], errors="coerce")

        source_atac.var["chrom"] = lifted_kept["chrom"].astype(str)
        source_atac.var["chromStart"] = lifted_kept["chromStart"].astype(int)
        source_atac.var["chromEnd"] = lifted_kept["chromEnd"].astype(int)

        # Update source assembly metadata now that intervals are in target assembly coordinates
        self.source_data.obs["assembly"] = self.target_data.obs["assembly"].unique()[0]

        # Write back updated ATAC modality
        if hasattr(self.source_data, "mod") and ("atac" in self.source_data.mod):
            self.source_data.mod["atac"] = source_atac
        else:
            self.source_data["atac"] = source_atac

#%%
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

#%% plot trained model's latents based on training source data

z_source_rna, _, z_source_atac, _, clip_embeddings_rna, clip_embeddings_atac = model.get_latent_representation(
    adata=model.adata,
    adata_atac=model.adata_atac,
    paired_data=True,
    counts_key="counts",
    adj_key="spatial_connectivities",
    cat_covariates_keys=None,
    only_active_gps=True,
    return_mu_std=True,
    separate_modalities=True,
    return_clip_embeddings=True,
    node_batch_size=model.node_batch_size_,
)

clip_embeddings_rna_magnitude = np.linalg.norm(clip_embeddings_rna, axis=1)
clip_embeddings_atac_magnitude = np.linalg.norm(clip_embeddings_atac, axis=1)
print(f'Mean magnitude of clip embeddings - RNA: {np.mean(clip_embeddings_rna_magnitude):.2f}, ATAC: {np.mean(clip_embeddings_atac_magnitude):.2f}')

# Normalize clip embeddings to unit norm
clip_embeddings_rna = clip_embeddings_rna / np.linalg.norm(clip_embeddings_rna, axis=1, keepdims=True)
clip_embeddings_atac = clip_embeddings_atac / np.linalg.norm(clip_embeddings_atac, axis=1, keepdims=True)
assert np.allclose(np.linalg.norm(clip_embeddings_rna, axis=1), 1), "RNA clip embeddings are not unit norm"
assert np.allclose(np.linalg.norm(clip_embeddings_atac, axis=1), 1), "ATAC clip embeddings are not unit norm"


clip_embeddings_adata = ad.AnnData(
    X=np.concatenate([clip_embeddings_rna, clip_embeddings_atac], axis=0),
    obs=pd.concat([
        model.adata.obs.assign(modality="rna"),
        model.adata_atac.obs.assign(modality="atac"),
    ], axis=0),
)

import scanpy as sc
sc.pp.pca(clip_embeddings_adata, n_comps=50)
sc.pp.neighbors(clip_embeddings_adata, use_rep='X_pca', n_neighbors=100)
sc.tl.umap(clip_embeddings_adata, min_dist=0.3)
sc.pl.umap(clip_embeddings_adata, color=['modality'], wspace=0.2)

#%%
data_aligner = DataAligner(
    source_data,
    target_data,
    source_assembly="mm10",
    target_assembly="mm39",
    source_name="Spatial_ATAC_RNA",
    target_name="EasySci_SLL"
)

data_aligner.find_gene_overlap()
data_aligner.find_peak_overlap()
data_aligner.align_features_by_overlap()

#%% TMP: artificial data setup
from scipy.sparse import csr_matrix

target_data = data_aligner.target_data

## RNA formatting
missing_genes = set(source_rna.var_names) - set(target_data['rna'].var_names)

dummy = ad.AnnData(
    X=csr_matrix((0, len(missing_genes))), 
    var=pd.DataFrame(index=list(missing_genes))
)

target_rna_with_missing = ad.concat([target_data['rna'], dummy], join="outer") # about 5.5 minutes
var_sort_idxs = target_rna_with_missing.var_names.get_indexer(source_rna.var_names)
target_rna_with_missing = target_rna_with_missing[:, var_sort_idxs]
assert target_rna_with_missing.var_names.equals(source_rna.var_names), "Target RNA data must have the same gene names"

# ensure GP names exist and reflect the model’s GP count/order
target_rna_with_missing.varm["nichecompass_gp_targets"] = source_rna.varm["nichecompass_gp_targets"].copy()
target_rna_with_missing.varm["nichecompass_gp_sources"] = source_rna.varm["nichecompass_gp_sources"].copy()

target_rna_with_missing.varm['nichecompass_gene_peaks'] = source_rna.varm['nichecompass_gene_peaks'].copy()
target_rna_with_missing.uns['nichecompass_genes_idx'] = source_rna.uns['nichecompass_genes_idx'].copy()
target_rna_with_missing.uns['nichecompass_target_genes_idx'] = source_rna.uns['nichecompass_target_genes_idx'].copy()
target_rna_with_missing.uns['nichecompass_source_genes_idx'] = source_rna.uns['nichecompass_source_genes_idx'].copy()
target_rna_with_missing.uns['nichecompass_gp_names'] = source_rna.uns['nichecompass_gp_names'].copy()

target_rna_with_missing.obsp['spatial_connectivities'] = target_rna.obsp['connectivities'].copy()

## ATAC formatting
#missing_peaks = set(data_aligner.source_data['atac'].var_names) - set(source_atac.var_names)
missing_peaks = set(model.adata_atac.var_names)

dummy = ad.AnnData(
    X=csr_matrix((model.adata_atac.n_obs, len(missing_peaks))), 
    var=pd.DataFrame(index=list(missing_peaks)),
    obs=pd.DataFrame(index=data_aligner.source_data['atac'].obs_names)
)

#target_atac_with_missing = ad.concat([target_data['atac'], dummy], join="outer")
#var_sort_idxs = target_atac_with_missing.var_names.get_indexer(data_aligner.source_data['atac'].var_names)
#target_atac_with_missing = target_atac_with_missing[:, var_sort_idxs]
#assert target_atac_with_missing.var_names.equals(data_aligner.source_data['atac'].var_names), "Target ATAC data must have the same peak names"

## Placeholder solution since can't use peak name mapper here
target_atac_with_missing = dummy.copy()

target_atac_with_missing.varm['nichecompass_ca_targets'] = model.adata_atac.varm['nichecompass_ca_targets'].copy()
target_atac_with_missing.varm['nichecompass_ca_sources'] = model.adata_atac.varm['nichecompass_ca_sources'].copy()

target_atac_with_missing.uns['nichecompass_peaks_idx'] = model.adata_atac.uns['nichecompass_peaks_idx'].copy()
target_atac_with_missing.uns['nichecompass_target_peaks_idx'] = model.adata_atac.uns['nichecompass_target_peaks_idx'].copy()
target_atac_with_missing.uns['nichecompass_source_peaks_idx'] = model.adata_atac.uns['nichecompass_source_peaks_idx'].copy()
#target_atac_with_missing.uns['nichecompass_gp_names'] = source_atac.uns['nichecompass_gp_names'].copy()

if ('spatial_connectivities' not in target_atac_with_missing.obsp) and ('connectivities' in model.adata_atac.obsp):
    target_atac_with_missing.obsp['spatial_connectivities'] = model.adata_atac.obsp['connectivities'].copy()
elif ('spatial_connectivities' not in target_atac_with_missing.obsp) and ('spatial_connectivities' in model.adata_atac.obsp):
    target_atac_with_missing.obsp['spatial_connectivities'] = model.adata_atac.obsp['spatial_connectivities'].copy()

## set RNA to the same number of cells as ATAC
target_rna_with_missing = target_rna_with_missing[:target_atac_with_missing.n_obs].copy()

#%%

model_target = CustomNicheCompass.load(
        dir_path=model_folder_path,
        adata=target_rna_with_missing,
        adata_atac=target_atac_with_missing,
        gp_names_key=gp_names_key,
    )
'''
model = CustomNicheCompass.load(
    dir_path=model_folder_path,
    adata=data_aligner.target_data['rna'],
    adata_atac=data_aligner.target_data['atac'],
    gp_names_key=gp_names_key
)
'''
z_target, _ = model_target.get_latent_representation(
            adata=model_target.adata,
            adata_atac=model_target.adata_atac,
            paired_data=False,
            counts_key="counts",
            adj_key="spatial_connectivities",
            cat_covariates_keys=None,
            only_active_gps=True,
            return_mu_std=True,
            node_batch_size=model_target.node_batch_size_,
)

#%% TMP: plot latent representation with umap

multimodal_obs = pd.concat([
    model_target.adata.obs.assign(modality="rna"),
    model_target.adata_atac.obs.assign(modality="atac"),
], axis=0)

latent_adata = ad.AnnData(
    X=z_target,
    obs=multimodal_obs,
)

import scanpy as sc
sc.pp.pca(latent_adata, n_comps=50)
sc.pp.neighbors(latent_adata, use_rep='X_pca', n_neighbors=100)
sc.tl.umap(latent_adata, min_dist=0.3)
sc.pl.umap(latent_adata, color=['modality'], wspace=0.2)
