#%% import libraries
import os
import tarfile
import re
import tempfile
import snapatac2 as snap
from tqdm import tqdm

#%% load data
datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/MouseDev_Spatial_Triomic'
tarfilename = 'GSE308623.tar'
tarpath = os.path.join(datapath, tarfilename)

with tarfile.open(tarpath, "r:*") as tar:
    filenames = tar.getnames()

developmental_atac_pattern = r'_P\d+S\d+_atac_fragments\.tsv\.gz$'
developmental_atac_files = sorted(f for f in filenames if re.search(developmental_atac_pattern, f))

#%% process data

with tarfile.open(tarpath, "r:*") as tar, tempfile.TemporaryDirectory() as tmp_dir:
    tmp_paths = []
    pbar = tqdm(developmental_atac_files)
    for file in pbar:
        sample_name = file.replace('.tsv.gz', '').split('_')[2]
        pbar.set_description(f'Extracting sample {sample_name}')

        # Extract to a temp directory so import_fragments can take file paths.
        tmp_path = os.path.join(tmp_dir, os.path.basename(file))
        with tar.extractfile(file) as f, open(tmp_path, "wb") as out_f:
            out_f.write(f.read())
        tmp_paths.append(tmp_path)

    adata = snap.pp.import_fragments(
        tmp_paths,
        chrom_sizes=snap.genome.mm10,  # Mouse genome reference
        file=None,  # Create in memory first
        min_num_fragments=200,
        sorted_by_barcode=False
    )

    #snap.pl.frag_size_distr(adata, interactive=False)
    snap.metrics.tsse(adata, snap.genome.mm10)
    #snap.pl.tsse(adata, interactive=False)

    #snap.pp.filter_cells(adata, min_counts=5000, min_tsse=10, max_counts=100000)
    snap.pp.add_tile_matrix(adata, bin_size=500)
    #snap.pp.select_features(adata)
    #snap.pp.scrublet(data)
    #snap.pp.filter_doublets(data)

# %%
