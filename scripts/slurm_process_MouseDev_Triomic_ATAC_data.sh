#!/bin/bash
#SBATCH --job-name=MouseDevTriomicATAC
#SBATCH --account=def-liyue
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=/home/dmannk/links/scratch/%x-%j.out
#SBATCH --error=/home/dmannk/links/scratch/%x-%j.err
#SBATCH --mail-user=dylan.mann-krzisnik@mail.mcgill.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Ensure relative paths in the Python script resolve correctly
cd /home/dmannk/links/projects/ctb-liyue/dmannk/BAKLAVAS_base/BAKLAVAS

source /home/dmannk/links/projects/ctb-liyue/dmannk/envs/snapatac_312/bin/activate
module load StdEnv/2023 python/3.12.4 arrow/22.0.0 rust/1.91.0

echo "JobID=${SLURM_JOB_ID:-}"
echo "Host=$(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
python -c "import os; print('affinity_cpus=', len(os.sched_getaffinity(0)))"

echo "SLURM_TMPDIR=${SLURM_TMPDIR:-}"
rsync -av /home/dmannk/links/scratch/GSM*h5ad "${SLURM_TMPDIR}/"
ls -lhtr "${SLURM_TMPDIR}/"

export PYTHONUNBUFFERED=1

# Run as a SLURM step so CPU binding is applied
srun --cpu-bind=cores -c "${SLURM_CPUS_PER_TASK:-1}" \
  python -u /home/dmannk/links/projects/ctb-liyue/dmannk/BAKLAVAS_base/BAKLAVAS/process_MouseDev_Triomic_ATAC_data.py


