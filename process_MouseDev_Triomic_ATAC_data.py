#%% load libraries
import os
import re
import numpy as np
import tarfile
import tempfile
import sys
from pathlib import Path
from tqdm import tqdm

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

#%% main function
def main() -> None:
    #datapath = os.path.abspath("../data/MouseDev_Spatial_Triomic")
    datapath = "/home/dmannk/links/projects/ctb-liyue/dmannk/BAKLAVAS_base/data/MouseDev_Spatial_Triomic"
    scratch_base = os.environ.get("SLURM_TMPDIR", "/home/dmannk/links/scratch")
    tarpath = os.path.join(datapath, "GSE308623.tar")

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
    gene_anno = '/home/dmannk/.cache/snapatac2/gencode.vM25.basic.annotation.gff3.gz'
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

    # spectral representation
    print(f"[PROGRESS] Selecting features...", flush=True)
    snap.pp.select_features(data, n_features=50000)
    print(f"[PROGRESS] Computing spectral representation...", flush=True)
    snap.tl.spectral(data)

    # UMAP
    #snap.tl.umap(data)
    #snap.pl.umap(data, color="sample", interactive=False)

    # Batch correction
    #snap.pp.mnc_correct(data, batch="sample")
    #snap.pp.harmony(data, batch="sample", max_iter_harmony=20)

    # UMAP after batch correction
    #snap.tl.umap(data, use_rep="X_spectral_mnn")
    #snap.pl.umap(data, color="sample", interactive=False)

    # UMAP after Harmony batch correction
    #snap.tl.umap(data, use_rep="X_spectral_harmony")
    #snap.pl.umap(data, color="sample", interactive=False)

    # Clustering
    #snap.pp.knn(data, use_rep="X_spectral_harmony")
    print(f"[PROGRESS] KNN...", flush=True)
    snap.pp.knn(data, use_rep="X_spectral")
    print(f"[PROGRESS] Leiden clustering...", flush=True)
    snap.tl.leiden(data)
    print(f"[PROGRESS] Leiden clustering completed.", flush=True)
    #snap.pl.umap(data, color="leiden", interactive=False)

    # Peak calling
    print(f"[PROGRESS] Peak calling...", flush=True)
    snap.tl.macs3(data, groupby='leiden', replicate='sample', n_jobs=min(_infer_n_jobs(default=8), 1))
    print(f"[PROGRESS] Merging peaks...", flush=True)
    merged_peaks = snap.tl.merge_peaks(data.uns['macs3'], chrom_sizes=snap.genome.mm10)
    print(f"Number of merged peaks: {merged_peaks.shape[0]}", flush=True)

    ## create peak matrix from merged peaks
    print(f"[PROGRESS] Creating peak matrix...", flush=True)
    peak_mat = snap.pp.make_peak_matrix(data, use_rep=merged_peaks['Peaks'])
    print(f"[PROGRESS] Peak matrix created successfully.", flush=True)

    ## save peak matrix to disk
    print(f"[PROGRESS] Saving peak matrix to disk...", flush=True)
    peak_mat.write_h5ad(os.path.join(datapath, "MouseDev_Triomic_ATAC_peak_matrix.h5ad"))
    print(f"[PROGRESS] Peak matrix saved successfully.", flush=True)

    # Save AnnDataSet to disk (writes the .h5ads file with all modifications)
    #print(f"[PROGRESS] Saving AnnDataSet to disk...")
    #adata = data.to_adata()

    # interferes with writing to disk, probably not enough memory to store all the data
    #adata.var = adata.var.drop(columns=['count'])
    #del adata.obsm['X_spectral']

    #adata.write_h5ad(os.path.join(scratch_base, "MouseDev_Triomic_ATAC.h5ad"))
    #adata.write_zarr(os.path.join(scratch_base, "MouseDev_Triomic_ATAC.zarr"))
    #print(f"[PROGRESS] AnnDataSet saved successfully.")

    # 6) Cleanup temp directory when you're done with everything (only if we created one)
    if workdir_ctx is not None:
        workdir_ctx.cleanup()


if __name__ == "__main__":
    main()
