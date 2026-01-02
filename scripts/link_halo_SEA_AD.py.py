#%% load libraries
import scanpy as sc
import os
from pathlib import Path
import matplotlib.pyplot as plt

from spatialdata import sanitize_table
from spatialdata_io.experimental import from_legacy_anndata
from spatialdata_io import image as spatialdata_image
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
import spatialdata_plot

from spatialdata.models import ShapesModel
import geopandas as gpd
from pyhaloxml import HaloXML

#%% load data
print('Loading data...')
datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD'

#spatial = sc.read_h5ad(os.path.join(datapath, 'merfish', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad'))
spatial = sc.read_h5ad(os.path.join(datapath, 'merfish', 'middle-temporal-gyrus', 'merfish_section_H21.33.021.Cx26.MTG.02.007.1.04.h5ad'))

sanitize_table(spatial)
spatial = from_legacy_anndata(spatial)

#%% load images
dapipath = os.path.join(datapath, 'merfish', 'middle-temporal-gyrus', '1170797664', 'DAPI_Max.tif')
dapi = spatialdata_image(dapipath, data_axes=['c', 'y','x'], coordinate_system=DEFAULT_COORDINATE_SYSTEM)

import tifffile
import numpy as np
import openslide
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

svspath = os.path.join(datapath, 'neuropathology', 'H21.33.021', 'H21.33.021-A7-ASYN', 'H21.33.021-A7-ASYN.svs')
# --- open slide ---
slide = openslide.OpenSlide(svspath)

# choose a pyramid level (0 = full res, higher = downsampled)
level = min(2, slide.level_count - 1)

# read full field-of-view at that level
img = slide.read_region(
    location=(0, 0),
    level=level,
    size=slide.level_dimensions[level],
)

# OpenSlide returns RGBA; drop alpha
img = np.array(img)[..., :3]   # (y, x, c)

# --- pixel size (microns per pixel) ---
mpp_x = float(slide.properties["openslide.mpp-x"])
mpp_y = float(slide.properties["openslide.mpp-y"])

# OpenSlide downsample factor
downsample = slide.level_downsamples[level]

# --- create SpatialData image ---
svs_image = Image2DModel.parse(
    img,
    dims=("y", "x", "c"),
    transformations={
        DEFAULT_COORDINATE_SYSTEM: Scale(
            [mpp_y * downsample, mpp_x * downsample],
            axes=("y", "x"),
        )
    },
)

spatial.images = {
    'dapi':dapi,
    'svs':svs_image
}

#%%
stains = ['A07-NEUN', 'A7-ASYN', 'A7-AT', 'A7-GFAP', 'A7-I6', 'A7-LFB']
donor = 'H21.33.021'

for stain in stains:
    print(f'Processing {stain}...')
    annotpath = Path(os.path.join(datapath, 'neuropathology', donor, f'{donor}-{stain}', f'{donor}-{stain}.annotations'))
    geojson_path = annotpath.with_suffix(".geojson")

    if not annotpath.exists():
        print(f'{annotpath.name} not found!')
        continue

    hx = HaloXML()
    hx.load(annotpath)
    hx.matchnegative()
    hx.to_geojson(geojson_path)

    gdf = gpd.read_file(geojson_path)
    stain_shapesmodel = ShapesModel.parse(gdf)

    stain_short = stain.split('-')[1]
    spatial.shapes[f'halo_layer_{stain_short}'] = stain_shapesmodel

#%%

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
spatial.pl.render_shapes('halo_layer_ASYN').pl.show(ax=ax[0])


# %%
