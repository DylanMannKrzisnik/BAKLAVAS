#%% load libraries
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="anndata" # suppress warnings about deprecated read functions
)

import os
import gc
import torch
import anndata
import numpy as np
import scanpy
import pandas as pd

from scvi.external import GIMVI
from lightning.pytorch.loggers import CSVLogger


def is_main_process():
    """Check if this is the main process in distributed training."""
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def cleanup_cuda():
    """Clean up CUDA memory and processes."""
    print('\nCleaning up CUDA resources...')
    
    # Force garbage collection
    gc.collect()
    
    # Empty CUDA cache on all devices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        
        # Print memory stats before cleanup
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i} memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated, '
                  f'{torch.cuda.memory_reserved(i) / 1024**3:.2f} GB reserved')
    
    print('CUDA cleanup complete.')

#%%
def main():
    #%% load data
    print('Loading data...')
    datapath = '/home/mcb/users/dmannk/BAKLAVA_base/data/SEA_AD'

    #rna = scanpy.read_h5ad(os.path.join(datapath, 'rna', 'rna_sub_100x.h5ad'))
    rna = scanpy.read_h5ad(os.path.join(datapath, 'rna', 'SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad'))
    #rna.layers['pseudo'] = rna.X.copy()
    rna.X = rna.layers['UMIs']

    #spatial = scanpy.read_h5ad(os.path.join(datapath, 'merfish', 'merfish_sub_100x.h5ad'))
    spatial = scanpy.read_h5ad(os.path.join(datapath, 'merfish', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad'))

    rna.obs = rna.obs.assign(modality='RNA')
    spatial.obs = spatial.obs.assign(modality='MERFISH')

    #%% filter data
    print('Filtering data...')
    gene_overlap = np.intersect1d(rna.var_names, spatial.var_names)
    print(f'Number of genes in overlap: {len(gene_overlap)}')

    rna = rna[:, rna.var_names.isin(gene_overlap)]
    spatial = spatial[:, spatial.var_names.isin(gene_overlap)]

    #%% split data into training and test sets
    print('Data setup...')
    # remove cells with no counts
    scanpy.pp.filter_cells(spatial, min_counts=10)
    scanpy.pp.filter_cells(rna, min_counts=10)

    # setup_anndata for spatial and sequencing data
    GIMVI.setup_anndata(spatial, labels_key="modality", batch_key="Donor ID")
    GIMVI.setup_anndata(rna, labels_key="modality", batch_key="Donor ID") # also have 'load_name'


    #%% setup_anndata for spatial and sequencing data
    print('Creating model...')

    # create our model
    model = GIMVI(rna, spatial)

    # create logger
    logpath = '/home/mcb/users/dmannk/BAKLAVA_base/outputs/logs'
    logger = CSVLogger(save_dir=logpath, name='gimVI_SEA_AD')
    print('logpath set to:', logpath)

    # train
    print('Training model...')
    model.train(
        max_epochs=100,
        train_size=0.9,     # without train_size, no validation set is used to compute metrics and no early stopping is performed
        validation_size=0.1,
        check_val_every_n_epoch=5,
        batch_size=2048,
        accelerator='gpu',
        devices=2,
        precision="16-mixed",   # 16-mixed precision for faster and lighter training (16-bit float)
        strategy="ddp_find_unused_parameters_true", # for notebook/interactive: ddp_notebook_find_unused_parameters_true
        early_stopping=False,  # Disabled for multi-GPU compatibility
        use_distributed_sampler=False, # incompatible with scvi
        datasplitter_kwargs={"num_workers": 0},
        logger=logger,
        plan_kwargs={
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "n_epochs_kl_warmup": 350
        }
    )
    
    # Synchronize all processes after training
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f'Process rank {torch.distributed.get_rank()} reached barrier after training')

    # Only run post-processing on the main process (rank 0) to avoid file conflicts
    if not is_main_process():
        print(f'\n=== Process rank {torch.distributed.get_rank()} skipping post-processing ===')
        return  # Exit early for non-main processes
    
    print('\n=== Running post-processing on main process (rank 0) ===')

    # Get the version number from the logger to create versioned output directories
    version_num = logger.version
    print(f'Training version: {version_num}')
    
    # Create versioned output directories
    base_output_dir = '/home/mcb/users/dmannk/BAKLAVA_base/outputs'
    version_output_dir = os.path.join(base_output_dir, 'gimVI_SEA_AD', f'version_{version_num}')
    version_figures_dir = os.path.join(version_output_dir, 'figures')
    version_models_dir = os.path.join(version_output_dir, 'models')
    dumps_output_dir = os.path.join(base_output_dir, 'dumps')

    os.makedirs(version_output_dir, exist_ok=True)
    os.makedirs(version_figures_dir, exist_ok=True)
    os.makedirs(version_models_dir, exist_ok=True)
    os.makedirs(dumps_output_dir, exist_ok=True)    # meant to only save latest results to not repeatedly save to disk (e.g. h5ad files)
    print(f'Created versioned output directories: {version_output_dir}, {version_figures_dir}, {version_models_dir}, {dumps_output_dir}')

    # Save model to versioned directory
    model_path = os.path.join(version_models_dir, 'gimVI_SEA_AD.pth')
    model.save(model_path, overwrite=True)
    print(f'Saved model to {model_path}')
    
    # Clear training-related GPU memory
    print('\nClearing training memory...')
    if hasattr(model, 'trainer') and model.trainer is not None:
        # Move model to CPU to free GPU memory
        model.module.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    #%% get latent representations
    print('Getting latent representations...')

    # get the latent representations for the sequencing and spatial data
    latent_rna, latent_spatial = model.get_latent_representation()

    # concatenate to one latent representation
    latent_representation = np.concatenate([latent_rna, latent_spatial])
    latent_adata = anndata.AnnData(latent_representation)

    # append Supertype to latent_adata
    latent_adata.obs['Supertype'] = pd.concat([rna.obs['Supertype'], spatial.obs['Supertype']]).values.astype(str)
    latent_adata.obs['Subclass'] = pd.concat([rna.obs['Subclass'], spatial.obs['Subclass']]).values.astype(str)

    # labels which cells were from the sequencing dataset and which were from the spatial dataset
    latent_labels = (["RNA"] * latent_rna.shape[0]) + (
        ["MERFISH"] * latent_spatial.shape[0]
    )
    latent_adata.obs["labels"] = latent_labels

    # Save full latent representations before UMAP
    latent_adata.write_h5ad(os.path.join(dumps_output_dir, 'latent_adata_full.h5ad'))
    print(f'Saved full latent representations to {dumps_output_dir}/latent_adata_full.h5ad')

    # subsample 100x
    print('Subsampling 100x...')
    latent_adata = latent_adata[::100].copy()
    print(f'After subsampling: {latent_adata.n_obs} cells')

    # compute umap
    print('Computing UMAP...')
    scanpy.pp.neighbors(latent_adata, use_rep="X")
    scanpy.tl.umap(latent_adata)
    print('UMAP computation complete')

    # Save subsampled latent representations after UMAP
    os.makedirs(dumps_output_dir, exist_ok=True)
    latent_adata.write_h5ad(os.path.join(dumps_output_dir, 'latent_adata_subsampled.h5ad'))
    print(f'Saved subsampled latent representations to {dumps_output_dir}/latent_adata_subsampled.h5ad')

    # plot and save UMAP to versioned figures directory
    print('Plotting UMAP and saving...')
    scanpy.settings.figdir = version_figures_dir
    scanpy.pl.umap(latent_adata, color=['labels', 'Subclass'], show=True, wspace=0.2, save='_gimVI_SEA_AD')
    print(f'Saved UMAP plots to {version_figures_dir}')
    
    print('\n=== Analysis complete ===')

#%%
if __name__ == '__main__':
    try:
        main()
        print('Done!')
    finally:
        # Always cleanup CUDA resources, even if there's an error
        cleanup_cuda()
        print('Script finished. All CUDA resources released.')
