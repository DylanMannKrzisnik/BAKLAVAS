#%% load libraries
import os
import re
import numpy as np
import tarfile
import tempfile
import sys
from pathlib import Path
from tqdm import tqdm
import polars as pl
import multiprocessing as mp

import snapatac2 as snap

def _tqdm(*args, **kwargs):
    """tqdm wrapper that writes progress bars to stdout (so SLURM --output captures them)."""
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    # Reduce log spam when running without a TTY (common for SLURM output files)
    kwargs.setdefault("mininterval", 5.0)
    kwargs.setdefault("maxinterval", 30.0)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)

def _infer_n_jobs(default: int = 8) -> int:
    """Infer a sensible worker count from SLURM / CPU affinity."""
    # Prefer SLURM allocation if present
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass

    # Fall back to actual CPU affinity (common when cpusets are enforced)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass

    # Last resort
    return max(1, int(os.cpu_count() or default))

os.environ["RUST_BACKTRACE"] = "1"

def save_ann_dataset(
    data: snap.AnnDataSet,
    sample_names: list,
    out_h5ad_paths: list,
    outpath: str,
    update_individual_files: bool = True,
    save_consolidated: bool = False,
) -> None:
    """
    Save AnnDataSet to disk with memory-efficient options.
    
    The .h5ads file is automatically saved and contains all modifications.
    This function optionally:
    1. Updates individual h5ad files with latest annotations (recommended)
    2. Saves a single consolidated file (memory-intensive)
    
    Parameters
    ----------
    data : snap.AnnDataSet
        The AnnDataSet to save
    sample_names : list
        List of sample names corresponding to individual AnnData objects
    out_h5ad_paths : list
        List of paths where individual h5ad files should be saved
    outpath : str
        Base directory for saving files
    update_individual_files : bool, default=True
        If True, update individual h5ad files with latest annotations from AnnDataSet.
        This ensures individual files have leiden clusters, UMAP coordinates, etc.
    save_consolidated : bool, default=False
        If True, save a single consolidated .zarr or .h5ad file (memory-intensive).
        Set via SAVE_CONSOLIDATED_FILE environment variable.
    """
    print(f"[INFO] AnnDataSet is already saved to: {data.filename}", flush=True)
    print(f"[INFO] This file contains all modifications and can be loaded without memory overhead.", flush=True)
    
    # Update individual h5ad files with latest annotations
    if update_individual_files:
        print(f"[PROGRESS] Updating individual h5ad files with latest annotations...", flush=True)
        import gc
        updated_count = 0
        failed_count = 0
        
        for name, out_path in _tqdm(zip(sample_names, out_h5ad_paths), 
                                      desc="Updating individual files", 
                                      total=len(sample_names)):
            try:
                # Extract sample-specific data from AnnDataSet
                # This loads only one sample's data at a time (memory-efficient)
                sample_mask = data.obs['sample'] == name
                if sample_mask.sum() == 0:
                    print(f"[WARNING] No cells found for sample {name}, skipping", flush=True)
                    failed_count += 1
                    continue
                
                # Create a view/copy of the sample data with all annotations
                # Note: This requires memory for one sample, but processes them sequentially
                sample_adata = data[sample_mask].to_adata()
                
                # Write the updated file
                sample_adata.write_h5ad(out_path, compression='gzip')
                updated_count += 1
                
                # Clean up to free memory before next iteration
                del sample_adata
                gc.collect()
                
            except MemoryError as e:
                print(f"[ERROR] Out of memory while processing sample {name}: {e}", flush=True)
                failed_count += 1
                gc.collect()
            except Exception as e:
                print(f"[WARNING] Error updating {name}: {e}", flush=True)
                failed_count += 1
                gc.collect()
        
        print(f"[PROGRESS] Updated {updated_count}/{len(sample_names)} individual h5ad files.", flush=True)
        if failed_count > 0:
            print(f"[WARNING] Failed to update {failed_count} files. The .h5ads file at {data.filename} still contains all data.", flush=True)
    else:
        print(f"[INFO] Skipping individual file updates (set update_individual_files=True to enable)", flush=True)
    
    # Optionally save consolidated file
    if save_consolidated:
        print(f"[PROGRESS] Converting AnnDataSet to single consolidated file (memory-intensive)...", flush=True)
        try:
            # Convert to AnnData (this loads data into memory)
            print(f"[INFO] Converting to AnnData (this may use significant memory)...", flush=True)
            adata = data.to_adata()
            
            # Drop large arrays that can be recomputed if needed
            # This reduces memory usage for the write operation
            if 'X_spectral' in adata.obsm:
                print(f"[INFO] Dropping X_spectral to save memory (can be recomputed)", flush=True)
                del adata.obsm['X_spectral']
            
            if 'count' in adata.var.columns:
                print(f"[INFO] Dropping 'count' column from var", flush=True)
                adata.var = adata.var.drop(columns=['count'])
            
            # Optionally also save as h5ad (uncomment if needed)
            h5ad_path = os.path.join(outpath, "MouseDev_Triomic_ATAC.h5ad")
            adata.write_h5ad(h5ad_path, compression='gzip')
            print(f"[PROGRESS] Saved as H5AD: {h5ad_path}", flush=True)

            # Write as Zarr (better for large datasets, supports chunked access)
            zarr_path = os.path.join(outpath, "MouseDev_Triomic_ATAC.zarr")
            print(f"[INFO] Writing to Zarr format (chunked, memory-efficient)...", flush=True)
            adata.write_zarr(zarr_path, chunks=(10000, None))
            print(f"[PROGRESS] Saved as Zarr: {zarr_path}", flush=True)
            
            
            # Clean up
            del adata
            import gc
            gc.collect()
            print(f"[INFO] Note: Large arrays (like X_spectral) remain in the .h5ads file", flush=True)
            
        except MemoryError as e:
            print(f"[WARNING] Not enough memory to create consolidated file: {e}", flush=True)
            print(f"[INFO] The .h5ads file at {data.filename} is sufficient for most use cases.", flush=True)
            print(f"[INFO] You can load it later with: snap.read_dataset('{data.filename}')", flush=True)
    else:
        print(f"[INFO] Skipping consolidated file conversion (set SAVE_CONSOLIDATED_FILE=true to enable)", flush=True)
        print(f"[INFO] The .h5ads file format is memory-efficient and recommended for large datasets.", flush=True)

#%% main function
def main() -> None:

    outpath = os.path.join("/home/dmannk/links/scratch", f"MouseDev_Triomic_ATAC_{os.environ.get('SLURM_JOB_ID', 'local')}")
    scratch_base = os.environ.get("SLURM_TMPDIR", outpath)
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(scratch_base, exist_ok=True)

    # Make *everything* use node-local temp, even if tempdir= is ignored (tl.macs3 bug #232)
    os.environ["TMPDIR"] = scratch_base
    tempfile.tempdir = scratch_base

    datapath = "/home/dmannk/links/projects/ctb-liyue/dmannk/BAKLAVAS_base/data/MouseDev_Spatial_Triomic"
    tarpath = os.path.join(datapath, "GSE308623.tar")
    gene_anno = '/home/dmannk/.cache/snapatac2/gencode.vM25.basic.annotation.gff3.gz'

    developmental_atac_pattern = r"_P\d+S\d+_atac_fragments\.tsv\.gz$"
    BIN_SIZE = 5000

    # Check if untarred directory exists (same name as tar minus .tar extension)
    untarred_dir = Path(datapath) / "GSE308623"
    use_untarred = untarred_dir.exists() and untarred_dir.is_dir()

    frag_paths = []
    out_h5ad_paths = []
    sample_names = []
    workdir_ctx = None  # Only create temp dir if we need to extract from tar

    if use_untarred:
        # Use fragment files directly from untarred directory
        print(f"Using untarred directory: {untarred_dir}", flush=True)
        fragment_files = sorted(
            [
                f
                for f in untarred_dir.iterdir()
                if f.is_file()
                and re.search(developmental_atac_pattern, f.name)
            ],
            key=lambda f: f.name,
        )

        for frag_file in _tqdm(fragment_files, desc="Collecting fragment files"):
            sample = frag_file.name.replace(".tsv.gz", "")
            #out_path = Path(datapath) / f"{sample}.h5ad"
            out_path = Path(scratch_base) / f"{sample}.h5ad"

            frag_paths.append(str(frag_file))
            out_h5ad_paths.append(str(out_path))
            sample_names.append(sample)
    else:
        # Fall back to extracting from tar
        print(f"Untarred directory not found, extracting from: {tarpath}", flush=True)
        # 1) Find matching members inside the tar
        with tarfile.open(tarpath, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile()
                and re.search(developmental_atac_pattern, os.path.basename(m.name))
            ]
        members = sorted(members, key=lambda m: os.path.basename(m.name))

        # 2) Extract all matching fragment files into a temp dir (ideally on node-local scratch)
        workdir_ctx = tempfile.TemporaryDirectory(dir=scratch_base)
        workdir = Path(workdir_ctx.name)

        with tarfile.open(tarpath, "r:*") as tar:
            for m in _tqdm(members, desc="Extracting fragment files"):
                base = os.path.basename(m.name)
                sample = base.replace(".tsv.gz", "")
                out_path = Path(datapath) / f"{sample}.h5ad"
                frag_path = workdir / base  # keep the .tsv.gz name

                # Stream member -> extracted gzip file on disk
                with tar.extractfile(m) as src, open(frag_path, "wb") as dst:
                    # chunked copy to avoid reading whole file into RAM
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(chunk)

                frag_paths.append(str(frag_path))
                out_h5ad_paths.append(str(out_path))
                sample_names.append(sample)

    # Check if all out_h5ad_paths already exist before running import_fragments
    all_exist = all(os.path.exists(p) for p in out_h5ad_paths)
    if all_exist:
        print(f"[INFO] All output h5ad files already exist. Skipping import_fragments.", flush=True)
        adatas = [snap.read(path, backed='r') for path in out_h5ad_paths]
    else:
        print(f"[INFO] Not all output h5ad files exist. Proceeding with import_fragments.", flush=True)
    
        print(f"[PROGRESS] Starting import_fragments for {len(frag_paths)} samples...", flush=True)
        adatas = snap.pp.import_fragments(
            frag_paths,
            file=out_h5ad_paths,  # writes anndata to file
            chrom_sizes=snap.genome.mm10,
            min_num_fragments=200,
            sorted_by_barcode=False,
            n_jobs=min(_infer_n_jobs(default=8), 8), # possible deadlock issues beyond 8 cores
        )
        print(f"[PROGRESS] import_fragments completed. Loaded {len(adatas)} AnnData objects.", flush=True)

    #data = snap.AnnDataSet(adatas=list(zip(sample_names, adatas)), filename=os.path.join(datapath, "MouseDev_Triomic_ATAC.h5ads"))

    # 4) Tutorial-style: these accept a list of AnnData
    gene_anno_exists = os.path.exists(gene_anno)

    print(f"[PROGRESS] Computing TSS enrichment scores...", flush=True)
    snap.metrics.tsse(adatas, gene_anno if gene_anno_exists else snap.genome.mm10)
    print(f"[PROGRESS] Filtering cells (min_tsse=1)...", flush=True)
    snap.pp.filter_cells(adatas, min_tsse=1)
    print(f"[PROGRESS] Adding tile matrix (bin_size={BIN_SIZE})...", flush=True)
    snap.pp.add_tile_matrix(adatas, bin_size=BIN_SIZE)
    print(f"[PROGRESS] Selecting features...", flush=True)
    snap.pp.select_features(adatas, n_features=None)
    # snap.pp.scrublet(adatas)
    # snap.pp.filter_doublets(adatas)

    # 5) Create AnnDataSet, like the tutorial
    print(f"[PROGRESS] Creating AnnDataSet...", flush=True)
    data = snap.AnnDataSet(
        adatas=[(name, adata) for name, adata in zip(sample_names, adatas)],
        filename=os.path.join(scratch_base, "MouseDev_Triomic_ATAC.h5ads"),
    )
    print(f"[PROGRESS] AnnDataSet created successfully.", flush=True)

    # Redo TSS enrichment scores on AnnDataSet
    #print(f"[PROGRESS] Redoing TSS enrichment scores on AnnDataSet...", flush=True)
    #snap.metrics.tsse(data, gene_anno if gene_anno_exists else snap.genome.mm10)
    
    # Generate plots
    #snap.pl.tsse(data, interactive=False, out_file=os.path.join(outpath, "MouseDev_Triomic_ATAC_tsse.png")) # RuntimeError: not found: n_fragment
    snap.pl.frag_size_distr(data, interactive=False, out_file=os.path.join(outpath, "MouseDev_Triomic_ATAC_frag_size_distr.png"))

    '''
    if os.path.exists(os.path.join(datapath, "MouseDev_Triomic_ATAC.h5ads")):
        data = snap.read_dataset(
            os.path.join(datapath, "MouseDev_Triomic_ATAC.h5ads"),
            adata_files_update=dict(zip(sample_names, out_h5ad_paths)),
            )
    '''

    # check for duplicate barcodes
    print(f"Number of cells: {data.n_obs}", flush=True)
    print(f"Number of unique barcodes: {np.unique(data.obs_names).size}", flush=True)

    unique_cell_ids = [sa + ":" + bc for sa, bc in zip(data.obs["sample"], data.obs_names)]
    data.obs_names = unique_cell_ids
    assert data.n_obs == np.unique(data.obs_names).size

    # add obs metadata
    data.obs['stage'] = data.obs['sample'].str.extract(r"_(P\d+)")
    data.obs['rep'] = data.obs['sample'].str.extract(r"(S\d+)")

    # spectral representation
    print(f"[PROGRESS] Selecting features...", flush=True)
    snap.pp.select_features(data, n_features=50000)
    print(f"[PROGRESS] Computing spectral representation...", flush=True)
    snap.tl.spectral(data)

    # Batch correction
    #snap.pp.mnc_correct(data, batch="stage")

    X_spectral_harmony = snap.pp.harmony(
        data,
        batch="sample",
        groupby="stage",
        use_rep="X_spectral",
        max_iter_harmony=10,
        max_iter_kmeans=10, # reduced for speed
        nclust=30, # reduced for speed
        theta=3,
        n_jobs=_infer_n_jobs(default=8),
        inplace=False, # if True: adata.obsm[use_rep + "_harmony"] = mat ~~~~ RuntimeError: dimension cannot be changed from 37221 to 30
    )
    data.obsm["X_spectral_harmony"] = np.ascontiguousarray(X_spectral_harmony, dtype=np.float64)

    # Clustering
    #snap.pp.knn(data, use_rep="X_spectral_harmony")
    print(f"[PROGRESS] KNN...", flush=True)
    snap.pp.knn(data, use_rep="X_spectral_harmony")
    print(f"[PROGRESS] Leiden clustering...", flush=True)
    snap.tl.leiden(data)
    print(f"[PROGRESS] Leiden clustering completed.", flush=True)

    # UMAP
    snap.tl.umap(data, use_rep="X_spectral_harmony", random_state=None if int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) > 1 else 42) # random_state seed removes parallelization
    #snap.pl.umap(data, color=["leiden", "sample", "stage", "rep"], interactive=False,
    snap.pl.umap(data, color=["leiden"], interactive=False,
        out_file=os.path.join(outpath, "MouseDev_Triomic_ATAC_UMAP.png"))

    # filter leiden clusters used for peak calling by number of cells
    leiden_stage_counts_threshold = 500
    data.obs["leiden_stage"] = data.obs["leiden"] + "_" + data.obs["stage"]
    counts = data.obs["leiden_stage"].value_counts()
    selected_leiden_stages = set(
        counts.filter(pl.col("count") >= leiden_stage_counts_threshold)["leiden_stage"].to_list()
    )
    print(f"Number of leiden clusters used for peak calling (n>={leiden_stage_counts_threshold}): {len(selected_leiden_stages)} out of {len(data.obs['leiden_stage'].unique())}", flush=True)

    # Peak calling
    print(f"[PROGRESS] Peak calling...", flush=True)

    snap.tl.macs3(
        data,
        groupby='leiden_stage',
        selections=selected_leiden_stages,
        qvalue=0.05,
        replicate=None, # 'None' means no replicates (only one group) - more akin to a union set of peaks
        replicate_qvalue=None, # only relevant if replicates are provided
        max_frag_size=200, # optional ATAC setting (keep nucleosome-free-ish)
        n_jobs=min(_infer_n_jobs(default=8), 8),
        tempdir=scratch_base
        )

    print(f"[PROGRESS] Merging peaks...", flush=True)
    merged_peaks = snap.tl.merge_peaks(data.uns['macs3'], chrom_sizes=snap.genome.mm10)
    print(f"Number of merged peaks: {merged_peaks.shape[0]}", flush=True)

    ## TMP: save data and merged peaks to disk
    merged_peaks.to_csv(os.path.join(outpath, "MouseDev_Triomic_ATAC_merged_peaks.csv"))

    save_ann_dataset(
        data=data,
        sample_names=sample_names,
        out_h5ad_paths=out_h5ad_paths,
        outpath=outpath,
        update_individual_files=False,
        save_consolidated=True,
    )

    ## create peak matrix from merged peaks
    print(f"[PROGRESS] Creating peak matrix...", flush=True)
    peak_mat = snap.pp.make_peak_matrix(data, use_rep=merged_peaks['Peaks'])
    print(f"[PROGRESS] Peak matrix created successfully.", flush=True)

    ## save peak matrix to disk
    print(f"[PROGRESS] Saving peak matrix to disk...", flush=True)
    peak_mat.write_h5ad(os.path.join(datapath, "MouseDev_Triomic_ATAC_peak_matrix.h5ad"))
    print(f"[PROGRESS] Peak matrix saved successfully.", flush=True)

    # Save AnnDataSet to disk (memory-efficient approach)
    '''
    save_consolidated = os.environ.get("SAVE_CONSOLIDATED_FILE", "false").lower() == "true"
    update_individual = os.environ.get("UPDATE_INDIVIDUAL_FILES", "true").lower() == "true"
    '''

    # 6) Cleanup temp directory when you're done with everything (only if we created one)
    if workdir_ctx is not None:
        workdir_ctx.cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # or "forkserver"
    import harmony_patch
    harmony_patch.apply()
    main()
