#!/bin/bash
# DECOMPOSE-targeted sphere phantom (A3) LUT: one task per random realisation.
#SBATCH --job-name=meso-a3
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/user/uqaste15/chisep-meso/slurm_logs/a3_%A_%a.log
set -euo pipefail
PY=${MESO_PY:-/scratch/user/uqaste15/envs/meso/bin/python}
HERE=${MESO_DIR:-/scratch/user/uqaste15/src/qsmf-wt-meso/prototypes/mesoscopic}
OUT=${MESO_OUT:-/scratch/user/uqaste15/chisep-meso/a3_parts}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
$PY $HERE/spheres_a3.py build --seed $SLURM_ARRAY_TASK_ID --out $OUT
