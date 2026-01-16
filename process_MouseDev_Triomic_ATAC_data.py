#%% import libraries
import os
import tarfile
import re
import tempfile
import shutil
import snapatac2 as snap
import anndata as ad
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
BIN_SIZE = 5000
adatas = []

with tarfile.open(tarpath, "r:*") as tar:
    pbar = tqdm(developmental_atac_files)
    for member in pbar:
        base = os.path.basename(member)
        sample_name = base.replace(".tsv.gz", "")
        pbar.set_description(f"Processing sample {sample_name}")

        scratch = os.environ.get("SLURM_TMPDIR", None)  # or set to a known larger path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv.gz", dir=scratch) as tmp_file:
            tmp_path = tmp_file.name
            with tar.extractfile(member) as f:
                shutil.copyfileobj(f, tmp_file)

        try:
            adata = snap.pp.import_fragments(
                tmp_path,
                chrom_sizes=snap.genome.mm10,
                file=None,
                min_num_fragments=200,
                sorted_by_barcode=False,
                # n_jobs=1,  # if your version supports it
            )
            snap.pp.add_tile_matrix(adata, bin_size=BIN_SIZE)
            adatas.append(adata)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        break


#%% process data, works directly on list of adatas

snap.metrics.tsse(adatas, snap.genome.mm10)
snap.pp.filter_cells(adatas, min_tsse=1)
snap.pp.add_tile_matrix(adatas, bin_size=BIN_SIZE)
snap.pp.select_features(adatas, n_features=None)
#snap.pp.scrublet(adatas)
#snap.pp.filter_doublets(adatas)

#%% create AnnDataSet

data = snap.AnnDataSet(
    adatas=[(filename.replace(".tsv.gz", "").split('_')[2], adata) for filename, adata in zip(developmental_atac_files, adatas)],
    filename="MouseDev_Triomic_ATAC.h5ads"
)
