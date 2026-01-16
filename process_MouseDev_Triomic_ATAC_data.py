import os
import re
import tarfile
import tempfile
from pathlib import Path

import snapatac2 as snap

datapath = "/home/mcb/users/dmannk/BAKLAVA_base/data/MouseDev_Spatial_Triomic"
tarpath = os.path.join(datapath, "GSE308623.tar")

developmental_atac_pattern = r"_P\d+S\d+_atac_fragments\.tsv\.gz$"
BIN_SIZE = 5000

# 1) Find matching members inside the tar
with tarfile.open(tarpath, "r:*") as tar:
    members = [
        m for m in tar.getmembers()
        if m.isfile() and re.search(developmental_atac_pattern, os.path.basename(m.name))
    ]
members = sorted(members, key=lambda m: os.path.basename(m.name))

# 2) Extract all matching fragment files into a temp dir (ideally on node-local scratch)
scratch_base = os.environ.get("SLURM_TMPDIR", None)
workdir_ctx = tempfile.TemporaryDirectory(dir=scratch_base)
workdir = Path(workdir_ctx.name)

frag_paths = []
out_h5ad_paths = []
sample_names = []

with tarfile.open(tarpath, "r:*") as tar:
    for m in members:
        base = os.path.basename(m.name)
        sample = base.replace(".tsv.gz", "")
        out_path = workdir / f"{sample}.h5ad"
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
    file=out_h5ad_paths,              # per-sample output, like the tutorial :contentReference[oaicite:2]{index=2}
    chrom_sizes=snap.genome.mm10,
    min_num_fragments=200,
    sorted_by_barcode=False,
)

# 4) Tutorial-style: these accept a list of AnnData :contentReference[oaicite:3]{index=3}
snap.metrics.tsse(adatas, snap.genome.mm10)
snap.pp.filter_cells(adatas, min_tsse=1)
snap.pp.add_tile_matrix(adatas, bin_size=BIN_SIZE)
snap.pp.select_features(adatas, n_features=None)

# 5) Create AnnDataSet, like the tutorial :contentReference[oaicite:4]{index=4}
data = snap.AnnDataSet(
    adatas=[(name, adata) for name, adata in zip(sample_names, adatas)],
    filename="MouseDev_Triomic_ATAC.h5ads",
)

# 6) Cleanup temp directory when you’re done with everything
workdir_ctx.cleanup()
