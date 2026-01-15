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

atac_adatas = []
pbar = tqdm(developmental_atac_files)
for file in pbar:

    sample_name = file.replace('.tsv.gz', '').split('_')[2]
    pbar.set_description(f'Processing sample {sample_name}')

    with tarfile.open(tarpath, "r:*") as tar:
        # Extract file to temporary location since import_fragments needs a file path
        tmp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tsv.gz') as tmp_file:
            tmp_path = tmp_file.name
            with tar.extractfile(file) as f:
                tmp_file.write(f.read())
        
        # Ensure temp file is closed before using it
        output_path = os.path.join(datapath, f'{sample_name}_atac.h5ad')
        
        # Remove output file if it already exists to avoid conflicts
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Create AnnData in memory first, then save to avoid file locking issues
        adata = snap.pp.import_fragments(
            tmp_path,
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
                
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    atac_adatas.append(adata)

# %%
