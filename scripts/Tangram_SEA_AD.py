#%% load libraries
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

rna = sc.read_h5ad(os.path.join(datapath, 'rna', 'SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad'))
#rna.layers['pseudo'] = rna.X.copy()
#rna.X = rna.layers['UMIs']

spatial = sc.read_h5ad(os.path.join(datapath, 'merfish', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad'))

#%% populate spatial library with DAPI image
def create_uns_library_merfish(adata, img=None):
    
    if img is None:
        H = int(adata.obsm["spatial"][:,1].max()) + 100
        W = int(adata.obsm["spatial"][:,0].max()) + 100
        blank = np.ones((H, W, 3), dtype=np.float32)

    adata.uns["spatial"] = {
        "merfish": {
            "images": {"hires": img if img is not None else blank},
            "scalefactors": {"tissue_hires_scalef": 1.0},
            "metadata": {"platform": "MERFISH"}
        }
    }
    return adata

dapipath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD/merfish/middle-temporal-gyrus/1170797659/DAPI_Max.tif'
dapi = sq.im.ImageContainer(dapipath)
spatial = create_uns_library_merfish(spatial, img=dapi)
sq.pl.spatial_scatter(spatial, shape=None, color=['Layer annotation','Subclass'], wspace=-0.2)

#%% [OPTIONAL for MERFISH] Segmentation and feature calculation

sq.im.process(img=dapi)
sq.im.segment(img=dapi, layer="image_smooth", method="watershed", channel=0)

# define image layer to use for segmentation
features_kwargs = {"segmentation": {"label_layer": "segmented_watershed", "props": ["label", "centroid"], "channels": [1, 2]}}

# calculate segmentation features
sq.im.calculate_image_features(spatial, dapi, layer="image", key_added="image_features", features_kwargs=features_kwargs, features="segmentation", mask_circle=True)

#%% find gene overlap and setup Tangram data

genes = list(set(rna.var_names) & set(spatial.var_names))
print(len(genes))

rna_count_per_spot = spatial.X.sum(axis=1).squeeze()
spatial.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)

mdata = mudata.MuData(
        {
            "sp_full": spatial,
            "sc_full": rna,
            "sp": spatial[:, genes].copy(),
            "sc": rna[:, genes].copy()
        }
    )

modalities = {"density_prior_key": "sp", "sc_layer": "sc", "sp_layer": "sp"}

Tangram.setup_mudata(
        mdata, density_prior_key="rna_count_based_density", modalities=modalities
    )

#%% train Tangram model

model = Tangram(mdata, constrained=False, target_count=False)
model.train(accelerator='cpu')

#%% project cell annotations and genes

mapper = model.get_mapper_matrix()
mdata.mod["sc"].obsm["tangram_mapper"] = mapper
labels = mdata.mod["sc"].obs['Subclass']
mdata.mod["sp"].obsm["tangram_ct_pred"] = model.project_cell_annotations(
    mdata.mod["sc"], mdata.mod["sp"], mapper, labels
)
mdata.mod["sp_sc_projection"] = model.project_genes(
    mdata.mod["sc"], mdata.mod["sp"], mapper
)
#mdata.mod["sp"].obs = mdata.mod["sp"].obs.join(mdata.mod["sp"].obsm["tangram_ct_pred"])
mdata.mod["sp"].obs = mdata.mod["sp"].obs.merge(mdata.mod["sp"].obsm["tangram_ct_pred"], left_index=True, right_index=True, suffixes=('', '_tangram'))

spatial.obs = mdata.mod["sp"].obs.copy()

#%% plot Tangram results

tangram_columns = spatial.obs.columns[spatial.obs.columns.str.contains('_tangram')]
sq.pl.spatial_scatter(spatial, shape=None, color=list(tangram_columns) + ['Subclass'], wspace=-0.3)

#%% load MERFISH data and DAPI image with SpatialData library
from spatialdata import sanitize_table
from spatialdata_io.experimental import from_legacy_anndata
from spatialdata_io import image as spatialdata_image
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
import spatialdata_plot

#spatial = make_anndata_spatialdata_compatible(spatial)
sanitize_table(spatial)
spatial = from_legacy_anndata(spatial)

dapipath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD/merfish/middle-temporal-gyrus/1170797659/DAPI_Max.tif'
dapi = spatialdata_image(dapipath, data_axes=['c', 'y','x'], coordinate_system=DEFAULT_COORDINATE_SYSTEM)
spatial.images = {'dapi':dapi}
spatial.pl.render_images("dapi", cmap='gray').pl.show()
