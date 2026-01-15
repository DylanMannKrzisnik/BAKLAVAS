import sys
import os
from datetime import datetime
import pandas as pd
import re

# 1. Define the Tee class to handle tqdm and stdout simultaneously
class Envilogger(object):
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        # Write to notebook as-is (keeps the progress bar moving)
        self.stderr.write(data)
        
        # Write to file: replace carriage returns with newlines to see loss history
        if data.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cleaned_data = data.replace('\r', '\n').strip()
            # Avoid writing duplicate empty newlines
            if cleaned_data:
                self.file.write(f"[{timestamp}] {cleaned_data}\n")
                self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.stderr.flush()
        self.file.flush()

# 2. Configuration and Directory Setup
def envi_train_with_logger(envi_model, output_path, train_params_dict):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. Execution with Redirection
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger = Envilogger(output_path)

    try:
        sys.stdout = logger
        sys.stderr = logger
        
        print(f"--- Starting ENVI Training Session: {datetime.now()} ---")
        
        # Only train the model - do NOT call impute_genes() or infer_niche_covet()
        envi_model.train(**train_params_dict)        
        print(f"--- Training Complete: {datetime.now()} ---")

    finally:
        # 4. Critical: Restore streams even if the code crashes
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.file.close()
        print(f"\n[Success] Full log saved to: {output_path}")

        return envi_model


def parse_envi_log(file_path):
    # Regex to match the timestamp and the four specific loss metrics
    # It looks for patterns like spatial: -1.264e+00
    pattern = re.compile(
        r"\[(?P<timestamp>.*?)\]\s+"
        r"spatial:\s+(?P<spatial>[-+0-9.e]+)\s+"
        r"sc:\s+(?P<sc>[-+0-9.e]+)\s+"
        r"cov:\s+(?P<cov>[-+0-9.e]+)\s+"
        r"kl:\s+(?P<kl>[-+0-9.e]+):"
    )

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # Convert the extracted strings to float/datetime
                row = match.groupdict()
                row['timestamp'] = pd.to_datetime(row['timestamp'])
                row['spatial'] = float(row['spatial'])
                row['sc'] = float(row['sc'])
                row['cov'] = float(row['cov'])
                row['kl'] = float(row['kl'])
                data.append(row)

    # Create DataFrame and remove duplicates (tqdm logs the same step twice sometimes)
    df = pd.DataFrame(data).drop_duplicates(subset=['spatial', 'sc', 'cov', 'kl'])
    return df.reset_index(drop=True)

def read_envi_predictions(st_dat, sc_dat, envi_model):
    st_dat.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
    st_dat.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
    st_dat.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
    st_dat.uns['COVET_genes'] =  envi_model.CovGenes
    st_dat.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']
    if 'cell_type_niche' in envi_model.spatial_data.obsm:
        st_dat.obsm['cell_type_niche'] = envi_model.spatial_data.obsm['cell_type_niche']

    sc_dat.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
    sc_dat.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
    sc_dat.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
    sc_dat.uns['COVET_genes'] =  envi_model.CovGenes
    if 'cell_type_niche' in envi_model.sc_data.obsm:
        sc_dat.obsm['cell_type_niche'] = envi_model.sc_data.obsm['cell_type_niche']

    return st_dat, sc_dat