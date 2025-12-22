# BAKLAVAS
Biological Alignment across K-Layered Latent Axes of Variation from Spatial data

## Project Structure

```
BAKLAVA_base/
├── BAKLAVA/              # Code repository
│   ├── scripts/          # Analysis scripts
│   │   ├── gimVI_SEA_AD.py           # Main gimVI training script
│   │   ├── plot_metrics.py           # Metrics visualization utilities
│   │   └── example_plot_metrics.py   # Example usage
│   └── README.md
├── data/                 # Input data files
└── outputs/              # Training outputs, figures, logs
    ├── figures/
    ├── logs/
    └── [model files]
```

## Usage

### Training

Run the gimVI model training:
```bash
cd /home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/scripts
python gimVI_SEA_AD.py
```

### Plotting Training Metrics

The `plot_metrics.py` module provides utilities for visualizing training metrics from CSV log files.

#### As an importable module:

```python
from plot_metrics import plot_training_metrics, plot_metric_comparison
import matplotlib.pyplot as plt

# Plot all metrics from a training run
fig, axes, df = plot_training_metrics(
    'path/to/metrics.csv',
    save_path='output_plot.png'  # Optional: omit to display instead
)

# Compare a specific metric across multiple runs
version_paths = ['version_1/metrics.csv', 'version_2/metrics.csv']
fig, ax = plot_metric_comparison(
    version_paths,
    metric_name='elbo_validation',
    labels=['Run 1', 'Run 2']
)
plt.show()
```

#### From command line:

```bash
# Display plots
python plot_metrics.py path/to/metrics.csv

# Save plots to file
python plot_metrics.py path/to/metrics.csv output_plot.png

# Example with actual path
python plot_metrics.py ../../outputs/logs/gimVI_SEA_AD/version_12/metrics.csv
```

#### Run the example script:

```bash
python example_plot_metrics.py
```

## Features

- **Multi-metric visualization**: Automatically plots all metrics in subplots
- **Metric comparison**: Compare specific metrics across multiple training runs
- **Flexible output**: Display interactively or save to file
- **Clean annotations**: Shows first and last values for each metric
- **Customizable**: Adjustable figure size, style, and layout
