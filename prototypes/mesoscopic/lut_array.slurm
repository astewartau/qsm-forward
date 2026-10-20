#!/bin/bash
# Build the mesoscopic LUT parts: one task per (geometry, seed); each task runs the
# static and the diffusion walk for that geometry.  5 geometries x 3 seeds = 15 tasks.
#SBATCH --job-name=meso-lut
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/user/uqaste15/chisep-meso/slurm_logs/lut_%A_%a.log
set -euo pipefail
PY=/home/uqaste15/miniconda3/envs/qsmxt/bin/python3
HERE=${MESO_DIR:-/scratch/user/uqaste15/src/qsmf-wt-meso/prototypes/mesoscopic}
OUT=${MESO_OUT:-/scratch/user/uqaste15/chisep-meso/lut_parts}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
GEOM=$((SLURM_ARRAY_TASK_ID % 5))
SEED=$((SLURM_ARRAY_TASK_ID / 5))
$PY $HERE/build_lut.py --geom $GEOM --seed $SEED --mode static --out $OUT
$PY $HERE/build_lut.py --geom $GEOM --seed $SEED --mode diffusion --out $OUT
