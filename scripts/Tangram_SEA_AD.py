import matplotlib.pyplot as plt
import os
import mudata
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scvi.external import Tangram

#%% load data
print('Loading data...')
datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD'

#rna = scanpy.read_h5ad(os.path.join(datapath, 'rna', 'rna_sub_100x.h5ad'))
rna = sc.read_h5ad(os.path.join(datapath, 'rna', 'SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad'))
#rna.layers['pseudo'] = rna.X.copy()
rna.X = rna.layers['UMIs']

#spatial = scanpy.read_h5ad(os.path.join(datapath, 'merfish', 'merfish_sub_100x.h5ad'))
spatial = sc.read_h5ad(os.path.join(datapath, 'merfish', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad'))
# %%
