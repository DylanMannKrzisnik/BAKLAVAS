#%% load libraries
import scanpy as sc
import os
from pathlib import Path

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
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

svspath = os.path.join(datapath, 'neuropathology', 'H21.33.021', 'H21.33.021-A7-ASYN', 'H21.33.021-A7-ASYN.svs')

# Open SVS file with tifffile - SVS files are pyramidal TIFFs
with tifffile.TiffFile(svspath) as tif:
    # SVS files have multiple series (resolution levels)
    # series[0] is the full resolution, series[1] is downsampled, etc.
    num_levels = len(tif.series)
    
    # Choose a level (0 = full res, 1 = first downsample, etc.)
    level = min(2, num_levels - 1)  # Level 2 or highest available
    
    # Get the series at this level
    series = tif.series[level]
    
    # Calculate downsample factor (based on dimensions relative to level 0)
    if level == 0:
        downsample = 1.0
    else:
        full_res_shape = tif.series[0].shape
        current_shape = series.shape
        downsample = full_res_shape[0] / current_shape[0]  # y dimension ratio
    
    print(f"Loading SVS level {level}/{num_levels-1}: {series.shape} (downsample: {downsample}x)")
    
    # Read the image at this level
    svs_array = series.asarray()
    
    # Handle different channel configurations
    if svs_array.ndim == 3:
        # Shape is (y, x, c) - need to transpose to (c, y, x)
        svs_array = np.transpose(svs_array, (2, 0, 1))
    elif svs_array.ndim == 2:
        # Grayscale - add channel dimension
        svs_array = svs_array[np.newaxis, :, :]
    
    # Remove alpha channel if present (4 channels -> 3 channels RGB)
    if svs_array.shape[0] == 4:
        svs_array = svs_array[:3, :, :]

# Create Image2DModel with proper transformation
# Scale transformation accounts for the downsampling
svs_image = Image2DModel.parse(
    svs_array,
    dims=("c", "y", "x"),
    transformations={DEFAULT_COORDINATE_SYSTEM: Scale([downsample, downsample], axes=("x", "y"))}
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

spatial.subset(['locations', 'halo_layer_ASYN']).pl.render_shapes().pl.show()

# %%
