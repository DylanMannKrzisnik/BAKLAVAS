#%% import libraries
import os
import tarfile
import re
import tempfile
import shutil
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
#
# Important notes:
# - Files *inside a tar* must be extracted to disk because snap.pp.import_fragments expects paths.
# - To avoid disk quota issues, process one sample at a time: extract -> import -> build matrix -> write -> cleanup.
# - bin_size=500 creates millions of features on mm10 and can explode memory; 5000 is a safer default.

BIN_SIZE = 5000

with tarfile.open(tarpath, "r:*") as tar:
    pbar = tqdm(developmental_atac_files)
    for file in pbar:
        sample_name = file.replace(".tsv.gz", "").split("_")[2]
        pbar.set_description(f"Processing sample {sample_name}")

        output_path = os.path.join(datapath, f"{sample_name}_atac.h5ad")
        if os.path.exists(output_path):
            os.remove(output_path)

        # Extract this one fragments file to a temporary path (streaming copy to avoid huge RAM spikes).
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv.gz") as tmp_file:
            tmp_path = tmp_file.name
            with tar.extractfile(file) as f:
                shutil.copyfileobj(f, tmp_file)

        try:
            adata = snap.pp.import_fragments(
                tmp_path,
                chrom_sizes=snap.genome.mm10,  # Mouse genome reference
                file=None,  # Create in memory
                min_num_fragments=200,
                sorted_by_barcode=False,
            )

            # Optional QC (doesn't create features)
            # snap.metrics.tsse(adata, snap.genome.mm10)

            # Create the actual counts/features matrix (cell × tiles) in adata.X
            snap.pp.add_tile_matrix(adata, bin_size=BIN_SIZE)

            # Persist and release memory/handles
            adata.write(output_path)
            del adata
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# %%
