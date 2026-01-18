import os
import re
import numpy as np
import tarfile
import tempfile
from pathlib import Path
from tqdm import tqdm

import snapatac2 as snap

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


def main() -> None:
    #datapath = os.path.abspath("../data/MouseDev_Spatial_Triomic")
    datapath = "/home/dmannk/links/projects/ctb-liyue/dmannk/BAKLAVAS_base/data/MouseDev_Spatial_Triomic"
    tarpath = os.path.join(datapath, "GSE308623.tar")

    developmental_atac_pattern = r"_P\d+S\d+_atac_fragments\.tsv\.gz$"
    BIN_SIZE = 5000

    # 1) Find matching members inside the tar
    with tarfile.open(tarpath, "r:*") as tar:
        members = [
            m for m in tar.getmembers()
            if m.isfile()
            and re.search(developmental_atac_pattern, os.path.basename(m.name))
        ]
    members = sorted(members, key=lambda m: os.path.basename(m.name))

    # 2) Extract all matching fragment files into a temp dir (ideally on node-local scratch)
    # scratch_base = os.environ.get("SLURM_TMPDIR", None)
    scratch_base = "/home/dmannk/links/scratch"
    workdir_ctx = tempfile.TemporaryDirectory(dir=scratch_base)
    workdir = Path(workdir_ctx.name)

    frag_paths = []
    out_h5ad_paths = []
    sample_names = []

    with tarfile.open(tarpath, "r:*") as tar:
        for m in tqdm(members, desc="Extracting fragment files"):
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

    # 3) Tutorial-style: one call that imports all samples
    adatas = snap.pp.import_fragments(
        frag_paths,
        file=out_h5ad_paths,  # writes anndata to file
        chrom_sizes=snap.genome.mm10,
        min_num_fragments=200,
        sorted_by_barcode=False,
        n_jobs=_infer_n_jobs(default=8),
    )

    # 4) Tutorial-style: these accept a list of AnnData
    snap.metrics.tsse(adatas, snap.genome.mm10)
    snap.pp.filter_cells(adatas, min_tsse=1)
    snap.pp.add_tile_matrix(adatas, bin_size=BIN_SIZE)
    snap.pp.select_features(adatas, n_features=None)
    # snap.pp.scrublet(adatas)
    # snap.pp.filter_doublets(adatas)

    # 5) Create AnnDataSet, like the tutorial
    data = snap.AnnDataSet(
        adatas=[(name, adata) for name, adata in zip(sample_names, adatas)],
        filename="MouseDev_Triomic_ATAC.h5ads",
    )

    # check for duplicate barcodes
    print(f"Number of cells: {data.n_obs}")
    print(f"Number of unique barcodes: {np.unique(data.obs_names).size}")

    unique_cell_ids = [sa + ":" + bc for sa, bc in zip(data.obs["sample"], data.obs_names)]
    data.obs_names = unique_cell_ids
    assert data.n_obs == np.unique(data.obs_names).size

    # spectral representation
    snap.tl.spectral(data)

    # UMAP
    snap.tl.umap(data)
    snap.pl.umap(data, color="sample", interactive=False)

    # Batch correction
    snap.pp.mnc_correct(data, batch="sample")
    snap.pp.harmony(data, batch="sample", max_iter_harmony=20)

    # UMAP after batch correction
    snap.tl.umap(data, use_rep="X_spectral_mnn")
    snap.pl.umap(data, color="sample", interactive=False)

    # UMAP after Harmony batch correction
    snap.tl.umap(data, use_rep="X_spectral_harmony")
    snap.pl.umap(data, color="sample", interactive=False)

    # Clustering
    snap.pp.knn(data, use_rep="X_spectral_harmony")
    snap.tl.leiden(data)
    snap.pl.umap(data, color="leiden", interactive=False)

    # 6) Cleanup temp directory when you’re done with everything
    workdir_ctx.cleanup()


if __name__ == "__main__":
    main()
