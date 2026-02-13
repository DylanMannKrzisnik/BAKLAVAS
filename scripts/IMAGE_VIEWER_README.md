# Image Viewer HTML Generation for HPO

## Overview

The HPO script now automatically generates HTML image viewers for common artifact patterns and logs them to the parent MLflow run. This allows you to view all trial artifacts (e.g., UMAP plots, loss curves) in a single page instead of opening individual tabs for each image. Each image is automatically labeled with its trial name (e.g., "trial_0001") for easy identification.

## Features

- **Automatic generation**: Image viewer HTML files are created after the HPO study completes
- **Trial name labeling**: Each image is labeled with its MLflow trial name (e.g., "trial_0001", "trial_0002") extracted automatically from the MLflow run metadata
- **Multiple patterns**: Generates viewers for common patterns:
  - `*target_holdout_umap_epoch_*.png` - UMAP visualizations at different epochs
  - `*loss_curves.png` - Training loss curves
  - `*embedding_*.png` - Embedding visualizations
- **Responsive grid layout**: Images are displayed in a responsive grid with hover effects
- **Logged as artifacts**: HTML files are logged to the parent MLflow run for easy access

## Usage

### Automatic (Default Behavior)

When you run the HPO script, image viewers are automatically generated:

```bash
python scripts/hpo_optuna_mouse_brain.py --experiment-name my_hpo_experiment
```

After the study completes, you'll see:
```
Generating image viewer HTML files...
  Logged: image_viewer_target_holdout_umap_epoch__png.html
  Logged: image_viewer_loss_curves_png.html
  Logged: image_viewer_embedding__png.html
```

### Accessing the Image Viewers

1. **Via MLflow UI**:
   - Start MLflow UI: `mlflow ui --backend-store-uri sqlite:////path/to/mlflow.db`
   - Navigate to your experiment and parent run
   - Click on the "Artifacts" tab
   - Download and open any `image_viewer_*.html` file in your browser

2. **Via File System**:
   - Image viewers are saved in: `mlflow_artifacts/<study_name>/image_viewer_*.html`
   - Open them directly in your browser or use a simple HTTP server:
     ```bash
     cd mlflow_artifacts/<study_name>
     python -m http.server 8000
     # Then navigate to http://localhost:8000/image_viewer_target_holdout_umap_epoch__png.html
     ```

## Example Output

Each image will display:
1. **Trial name** (bold, dark blue): e.g., "trial_0001", "trial_0042"
2. **Image path** (small, monospace): e.g., "./a6e99895149647e9a174ac9a880c076f/artifacts/umap/target_holdout_umap_epoch_5.png"
3. **The image itself** with lazy loading for performance

## Custom Patterns

To generate image viewers for custom patterns, you can use the utility functions directly:

```python
from hpo_mouse_brain_utils import generate_image_viewer_html

# Generate viewer for a specific pattern with trial names
html_path = generate_image_viewer_html(
    base_dir="/path/to/artifact/dir",
    pattern="*my_custom_plot*.png",
    output_filename="custom_viewer.html",
    title="My Custom Image Viewer",
    parent_run_id="your_parent_run_id",  # Required for trial name extraction
)

# Log to MLflow
mlflow.log_artifact(html_path)
```

Or generate multiple viewers at once:

```python
from hpo_mouse_brain_utils import generate_image_viewers_for_study

# Generate viewers for multiple patterns
patterns = [
    "*target_holdout_umap_epoch_*.png",
    "*loss_curves.png",
    "*custom_plot*.png",
]

html_files = generate_image_viewers_for_study(
    artifact_dir="/path/to/artifact/dir",
    patterns=patterns,
    parent_run_id="your_parent_run_id",  # Required for trial name extraction
)

# Log all to MLflow
for html_file in html_files:
    mlflow.log_artifact(html_file)
```

## Implementation Details

### New Functions in `hpo_mouse_brain_utils.py`

1. **`get_trial_name_mapping(parent_run_id)`**
   - Queries MLflow to get a mapping of run_id to run_name for all child runs
   - Returns a dictionary like `{"abc123...": "trial_0001", ...}`

2. **`extract_run_id_from_path(image_path)`**
   - Extracts the MLflow run ID from an image path
   - Looks for a 32-character hex string in the path components

3. **`find_images_by_pattern(base_dir, pattern)`**
   - Finds all images matching a glob pattern in the specified directory
   - Returns a sorted list of relative paths

4. **`generate_image_viewer_html(base_dir, pattern, output_filename, title, parent_run_id)`**
   - Generates an HTML file with a responsive grid of images
   - Extracts and displays trial names for each image
   - Returns the absolute path to the generated HTML file

5. **`generate_image_viewers_for_study(artifact_dir, patterns, parent_run_id)`**
   - Convenience function to generate viewers for multiple patterns
   - Returns a list of paths to generated HTML files

### Integration Points

The image viewer generation is integrated at two points in `hpo_optuna_mouse_brain.py`:

1. **Launcher mode** (multi-GPU): After all workers complete
2. **Single worker mode**: After the study completes

Both integration points:
- Pass the `parent_run_id` to enable trial name extraction
- Log the generated HTML files as artifacts to the parent MLflow run

### Trial Name Extraction Process

1. When generating an image viewer, the `parent_run_id` is provided
2. The function queries MLflow to get all child runs of the parent
3. A mapping of `run_id` to `run_name` is created
4. For each image path, the run ID is extracted (the 32-char hex string in the path)
5. The run ID is looked up in the mapping to get the trial name
6. The trial name is displayed prominently above each image

## HTML Output Structure

The generated HTML includes:
- A title with the image pattern
- Total count of images found
- A responsive grid layout (400px minimum per image)
- Each image container shows:
  - **Trial name** (bold, 14px, dark blue) - e.g., "trial_0001"
  - **Image path** (small, 12px, gray, monospace) - the relative path
  - **The image** with responsive sizing
- Hover effects for better UX
- Lazy loading for performance

## Troubleshooting

**No trial names displayed**: If images don't show trial names, check that:
- The `parent_run_id` is being passed correctly
- The MLflow tracking server is accessible
- The parent run exists and has child runs

**No images found**: If you see "Warning: No images found matching pattern", check that:
- The pattern matches actual files in the artifact directory
- The artifact directory path is correct
- Images were successfully generated by the trials

**Import errors**: Make sure you're running the script in the correct conda environment with all dependencies installed.

**MLflow client errors**: Ensure your MLflow tracking URI is set correctly and the database is accessible.
