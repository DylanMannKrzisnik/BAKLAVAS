import pandas as pd
import numpy as np
import scanpy as sc

def multimodal_latents_adata(modality1_dict, modality2_dict, latent_key):

    adata1 = list(modality1_dict.values())[0]
    adata2 = list(modality2_dict.values())[0]
    modality1_name = list(modality1_dict.keys())[0]
    modality2_name = list(modality2_dict.keys())[0]

    obs = pd.concat([
        adata1.obs.assign(modality = modality1_name),
        adata2.obs.assign(modality = modality2_name)
        ], axis = 0)

    latents = np.concatenate([
        adata1.obsm[latent_key],
        adata2.obsm[latent_key]
        ], axis = 0)

    adata = sc.AnnData(
        X = latents,
        obs = obs
    )

    return adata