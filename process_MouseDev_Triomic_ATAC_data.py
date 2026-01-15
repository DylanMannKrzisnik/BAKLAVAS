import os
import tarfile
import re
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

rna_adatas = []
pbar = tqdm(developmental_atac_files)
for file in pbar:

    sample_name = file.strip('.csv.gz').split('_')[2]
    pbar.set_description(f'Processing sample {sample_name}')

    with tarfile.open(tarpath, "r:*") as tar:
        f = tar.extractfile(file)
        
        adata = snap.pp.import_fragments(
            f,
            chrom_sizes=snap.genome.hg38,  # or a dict of {chr: size}
            file=os.path.join(datapath, f'{sample_name}_atac.h5ad'),
            min_num_fragments=200
        )

    break