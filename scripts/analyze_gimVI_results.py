#!/usr/bin/env python3
"""
Script to load and analyze gimVI_SEA_AD training results.

This script loads:
- Training metrics from CSV logs
- Saved latent representations
- Optional: trained model for further analysis

And generates comprehensive visualizations and analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata

# Import plotting utilities from plot_metrics.py
from plot_metrics import plot_training_metrics, plot_metric_comparison


def find_latest_version(base_log_dir):
    """Find the latest version directory in the logs."""
    log_path = Path(base_log_dir)
    version_dirs = sorted(log_path.glob('version_*'))
    if not version_dirs:
        raise ValueError(f"No version directories found in {base_log_dir}")
    return version_dirs[-1]


def load_training_metrics(metrics_path):
    """Load training metrics from CSV file."""
    print(f'Loading training metrics from: {metrics_path}')
    df = pd.read_csv(metrics_path)
    print(f'Loaded {len(df)} rows of metrics')
    print(f'Available metrics: {[col for col in df.columns if col not in ["epoch", "step"]]}')
    return df


def load_latent_representations(latent_path):
    """Load latent representations from h5ad file."""
    print(f'Loading latent representations from: {latent_path}')
    adata = sc.read_h5ad(latent_path)
    print(f'Loaded latent representation: {adata.n_obs} cells x {adata.n_vars} latent dimensions')
    print(f'Available obs keys: {list(adata.obs.keys())}')
    return adata


def plot_latent_analysis(latent_adata, save_dir, figsize=(20, 15)):
    """Generate comprehensive analysis plots of latent representations."""
    print('\nGenerating latent representation analysis plots...')
    
    # Set up figure directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Basic UMAP plots
    print('Plotting UMAP representations...')
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # UMAP by modality
    if 'X_umap' in latent_adata.obsm:
        sc.pl.umap(latent_adata, color='labels', ax=axes[0, 0], show=False, title='UMAP by Modality')
        sc.pl.umap(latent_adata, color='Supertype', ax=axes[0, 1], show=False, 
                  title='UMAP by Supertype', legend_loc='on data', legend_fontsize=6)
        sc.pl.umap(latent_adata, color='Subclass', ax=axes[1, 0], show=False,
                  title='UMAP by Subclass', legend_loc='right margin', legend_fontsize=6)
        
        # UMAP colored by first latent dimension
        axes[1, 1].scatter(latent_adata.obsm['X_umap'][:, 0], 
                          latent_adata.obsm['X_umap'][:, 1],
                          c=latent_adata.X[:, 0], cmap='viridis', s=1, alpha=0.5)
        axes[1, 1].set_xlabel('UMAP1')
        axes[1, 1].set_ylabel('UMAP2')
        axes[1, 1].set_title('UMAP colored by 1st latent dimension')
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Latent dim 1')
    
    plt.tight_layout()
    umap_path = save_dir / 'latent_umap_analysis.png'
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    print(f'Saved UMAP analysis to: {umap_path}')
    plt.close()
    
    # 2. Latent dimension statistics
    print('Analyzing latent dimension statistics...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of latent values
    axes[0, 0].hist(latent_adata.X.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Latent value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of all latent values')
    axes[0, 0].set_yscale('log')
    
    # Mean and std per dimension
    means = latent_adata.X.mean(axis=0)
    stds = latent_adata.X.std(axis=0)
    dims = np.arange(len(means))
    
    axes[0, 1].bar(dims, means, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Latent dimension')
    axes[0, 1].set_ylabel('Mean value')
    axes[0, 1].set_title('Mean value per latent dimension')
    
    axes[1, 0].bar(dims, stds, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Latent dimension')
    axes[1, 0].set_ylabel('Standard deviation')
    axes[1, 0].set_title('Std deviation per latent dimension')
    
    # Correlation heatmap of first 20 dimensions
    n_dims_to_show = min(20, latent_adata.n_vars)
    corr = np.corrcoef(latent_adata.X[:, :n_dims_to_show].T)
    im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1, 1].set_xlabel(f'Latent dimension (first {n_dims_to_show})')
    axes[1, 1].set_ylabel(f'Latent dimension (first {n_dims_to_show})')
    axes[1, 1].set_title('Correlation between latent dimensions')
    plt.colorbar(im, ax=axes[1, 1], label='Correlation')
    
    plt.tight_layout()
    stats_path = save_dir / 'latent_statistics.png'
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f'Saved latent statistics to: {stats_path}')
    plt.close()
    
    # 3. Modality integration analysis
    print('Analyzing modality integration...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Separate RNA and MERFISH
    rna_mask = latent_adata.obs['labels'] == 'RNA'
    merfish_mask = latent_adata.obs['labels'] == 'MERFISH'
    
    # Compare latent distributions by modality
    for i in range(min(3, latent_adata.n_vars)):
        axes[i].hist(latent_adata.X[rna_mask, i], bins=50, alpha=0.5, 
                    label='RNA', density=True, edgecolor='black')
        axes[i].hist(latent_adata.X[merfish_mask, i], bins=50, alpha=0.5, 
                    label='MERFISH', density=True, edgecolor='black')
        axes[i].set_xlabel(f'Latent dimension {i+1}')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Distribution comparison (dim {i+1})')
        axes[i].legend()
    
    plt.tight_layout()
    modality_path = save_dir / 'modality_integration.png'
    plt.savefig(modality_path, dpi=300, bbox_inches='tight')
    print(f'Saved modality integration to: {modality_path}')
    plt.close()
    
    # 4. Cell type analysis
    print('Analyzing cell type representation...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count cells per supertype
    supertype_counts = latent_adata.obs['Supertype'].value_counts()
    axes[0].barh(range(len(supertype_counts)), supertype_counts.values, 
                alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(supertype_counts)))
    axes[0].set_yticklabels(supertype_counts.index, fontsize=8)
    axes[0].set_xlabel('Number of cells')
    axes[0].set_title('Cells per Supertype')
    axes[0].invert_yaxis()
    
    # Count cells per subclass (top 20)
    subclass_counts = latent_adata.obs['Subclass'].value_counts().head(20)
    axes[1].barh(range(len(subclass_counts)), subclass_counts.values,
                alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_yticks(range(len(subclass_counts)))
    axes[1].set_yticklabels(subclass_counts.index, fontsize=8)
    axes[1].set_xlabel('Number of cells')
    axes[1].set_title('Cells per Subclass (top 20)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    celltype_path = save_dir / 'celltype_counts.png'
    plt.savefig(celltype_path, dpi=300, bbox_inches='tight')
    print(f'Saved cell type counts to: {celltype_path}')
    plt.close()
    
    print('Latent analysis complete!')


def generate_summary_report(metrics_df, latent_adata, output_path):
    """Generate a text summary report of the analysis."""
    print('\nGenerating summary report...')
    
    with open(output_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('gimVI SEA-AD Analysis Summary Report\n')
        f.write('=' * 80 + '\n\n')
        
        # Training metrics summary
        f.write('TRAINING METRICS SUMMARY\n')
        f.write('-' * 80 + '\n')
        f.write(f'Total epochs: {metrics_df["epoch"].max():.0f}\n')
        
        for col in metrics_df.columns:
            if col not in ['epoch', 'step']:
                non_nan = metrics_df[col].dropna()
                if len(non_nan) > 0:
                    f.write(f'\n{col}:\n')
                    f.write(f'  Initial: {non_nan.iloc[0]:.4f}\n')
                    f.write(f'  Final: {non_nan.iloc[-1]:.4f}\n')
                    f.write(f'  Best: {non_nan.min():.4f} (epoch {metrics_df.loc[non_nan.idxmin(), "epoch"]:.0f})\n')
        
        # Latent representation summary
        f.write('\n\nLATENT REPRESENTATION SUMMARY\n')
        f.write('-' * 80 + '\n')
        f.write(f'Total cells: {latent_adata.n_obs:,}\n')
        f.write(f'Latent dimensions: {latent_adata.n_vars}\n')
        
        f.write('\nCells by modality:\n')
        for modality, count in latent_adata.obs['labels'].value_counts().items():
            f.write(f'  {modality}: {count:,} ({count/latent_adata.n_obs*100:.1f}%)\n')
        
        f.write('\nCells by Supertype:\n')
        for supertype, count in latent_adata.obs['Supertype'].value_counts().head(10).items():
            f.write(f'  {supertype}: {count:,}\n')
        
        f.write('\nLatent space statistics:\n')
        f.write(f'  Mean: {latent_adata.X.mean():.4f}\n')
        f.write(f'  Std: {latent_adata.X.std():.4f}\n')
        f.write(f'  Min: {latent_adata.X.min():.4f}\n')
        f.write(f'  Max: {latent_adata.X.max():.4f}\n')
        
        f.write('\n' + '=' * 80 + '\n')
    
    print(f'Summary report saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze gimVI SEA-AD training results and latent representations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version', type=int, default=None,
                       help='Version number to analyze (default: latest)')
    parser.add_argument('--base-dir', type=str, 
                       default='/home/mcb/users/dmannk/BAKLAVA_base/outputs',
                       help='Base output directory')
    parser.add_argument('--log-dir', type=str,
                       default='/home/mcb/users/dmannk/BAKLAVA_base/outputs/logs/gimVI_SEA_AD',
                       help='Directory containing training logs')
    parser.add_argument('--dumps-dir', type=str,
                       default='/home/mcb/users/dmannk/BAKLAVA_base/outputs/dumps',
                       help='Directory containing saved latent representations')
    parser.add_argument('--use-full', action='store_true',
                       help='Use full latent representations instead of subsampled')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis plots (default: version-specific)')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively instead of just saving')
    #args = parser.parse_known_args()[0]
    args = parser.parse_args()
    
    print('=' * 80)
    print('gimVI SEA-AD Results Analysis')
    print('=' * 80)
    
    # Determine which version to analyze
    if args.version is not None:
        version_name = f'version_{args.version}'
        metrics_path = Path(args.log_dir) / version_name / 'metrics.csv'
    else:
        # Find latest version
        latest_version_dir = find_latest_version(args.log_dir)
        version_name = latest_version_dir.name
        metrics_path = latest_version_dir / 'metrics.csv'
        print(f'Using latest version: {version_name}')
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.base_dir) / 'gimVI_SEA_AD' / version_name / 'analysis'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir}')
    
    # Load training metrics
    print('\n' + '-' * 80)
    print('LOADING TRAINING METRICS')
    print('-' * 80)
    metrics_df = load_training_metrics(metrics_path)
    
    # Plot training metrics
    print('\nPlotting training metrics...')
    fig, axes, _ = plot_training_metrics(
        metrics_path, 
        save_path=output_dir / 'training_metrics.png',
        figsize=(16, 12)
    )
    if args.show:
        plt.show()
    else:
        plt.close()
    
    # Load latent representations
    print('\n' + '-' * 80)
    print('LOADING LATENT REPRESENTATIONS')
    print('-' * 80)
    
    latent_filename = 'latent_adata_full.h5ad' if args.use_full else 'latent_adata_subsampled.h5ad'
    latent_path = Path(args.dumps_dir) / latent_filename
    
    if not latent_path.exists():
        print(f'Warning: {latent_path} not found!')
        print('Available files in dumps directory:')
        for f in Path(args.dumps_dir).glob('*.h5ad'):
            print(f'  - {f.name}')
    
    latent_adata = load_latent_representations(latent_path)
    
    sc.tl.embedding_density(latent_adata, groupby='labels')
    sc.pl.embedding_density(latent_adata, key='umap_density_labels')

if __name__ == '__main__':
    main()

