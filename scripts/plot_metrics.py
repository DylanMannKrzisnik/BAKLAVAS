"""
Utility functions for plotting training metrics from CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_metrics(metrics_csv_path, save_path=None, figsize=(15, 10), style='darkgrid'):
    """
    Read a metrics.csv file and plot all metrics per epoch.
    
    Parameters
    ----------
    metrics_csv_path : str or Path
        Path to the metrics.csv file (e.g., 'outputs/logs/gimVI_SEA_AD/version_12/metrics.csv')
    save_path : str or Path, optional
        If provided, save the figure to this path. Otherwise, display with plt.show()
    figsize : tuple, optional
        Figure size as (width, height). Default: (15, 10)
    style : str, optional
        Seaborn style to use. Default: 'darkgrid'
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    axes : array of matplotlib.axes.Axes
        Array of axes objects for further customization
    df : pandas.DataFrame
        The loaded metrics dataframe
    
    Examples
    --------
    >>> fig, axes, df = plot_training_metrics('outputs/logs/gimVI_SEA_AD/version_12/metrics.csv')
    >>> plt.show()
    
    >>> # Save to file instead of displaying
    >>> plot_training_metrics('metrics.csv', save_path='training_metrics.png')
    """
    # Set seaborn style
    sns.set_style(style)
    
    # Read the metrics CSV
    df = pd.read_csv(metrics_csv_path)
    
    # Get all metric columns (exclude 'epoch' and 'step')
    metric_columns = [col for col in df.columns if col not in ['epoch', 'step']]
    
    # Determine number of subplots needed
    n_metrics = len(metric_columns)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f'Training Metrics - {Path(metrics_csv_path).parent.name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metric_columns):
        ax = axes[idx]
        
        # Remove any NaN values for cleaner plotting
        plot_data = df[['epoch', metric]].dropna()
        
        # Plot the metric
        ax.plot(plot_data['epoch'], plot_data[metric], 
                linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add value annotations for first and last points
        if len(plot_data) > 0:
            first_val = plot_data[metric].iloc[0]
            last_val = plot_data[metric].iloc[-1]
            first_epoch = plot_data['epoch'].iloc[0]
            last_epoch = plot_data['epoch'].iloc[-1]
            
            ax.annotate(f'{first_val:.2f}', 
                       xy=(first_epoch, first_val),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
            ax.annotate(f'{last_val:.2f}', 
                       xy=(last_epoch, last_val),
                       xytext=(5, -15), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')
    
    return fig, axes, df


def plot_metric_comparison(metrics_csv_paths, metric_name='elbo_validation', 
                          labels=None, save_path=None, figsize=(12, 6)):
    """
    Compare a specific metric across multiple training runs.
    
    Parameters
    ----------
    metrics_csv_paths : list of str or Path
        List of paths to metrics.csv files to compare
    metric_name : str, optional
        Name of the metric to compare. Default: 'elbo_validation'
    labels : list of str, optional
        Labels for each run. If None, uses parent directory names
    save_path : str or Path, optional
        If provided, save the figure to this path
    figsize : tuple, optional
        Figure size. Default: (12, 6)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> paths = ['logs/version_1/metrics.csv', 'logs/version_2/metrics.csv']
    >>> fig, ax = plot_metric_comparison(paths, metric_name='elbo_validation')
    >>> plt.show()
    """
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate labels if not provided
    if labels is None:
        labels = [Path(p).parent.name for p in metrics_csv_paths]
    
    # Plot each run
    for path, label in zip(metrics_csv_paths, labels):
        df = pd.read_csv(path)
        plot_data = df[['epoch', metric_name]].dropna()
        ax.plot(plot_data['epoch'], plot_data[metric_name], 
               linewidth=2, marker='o', markersize=3, alpha=0.7, label=label)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')
    
    return fig, ax


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
        save_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f'Plotting metrics from: {metrics_path}')
        fig, axes, df = plot_training_metrics(metrics_path, save_path=save_path)
        
        if save_path is None:
            plt.show()
    else:
        print("Usage: python plot_metrics.py <path_to_metrics.csv> [save_path]")
        print("\nExample:")
        print("  python plot_metrics.py outputs/logs/gimVI_SEA_AD/version_12/metrics.csv")
        print("  python plot_metrics.py outputs/logs/gimVI_SEA_AD/version_12/metrics.csv training_plot.png")

