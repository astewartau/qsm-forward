#!/bin/bash
# Iron calibration sweep: one task per (population mix, seed). Mix list in $MESO_CALIB_LIST
# (one --pops spec per line); $MESO_NSEED seeds each (default 3).
#SBATCH --job-name=meso-calib
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/uqaste15/chisep-meso/slurm_logs/calib_%A_%a.log
set -euo pipefail
PY=${MESO_PY:-/scratch/user/uqaste15/envs/meso/bin/python}
HERE=${MESO_DIR:-/scratch/user/uqaste15/src/qsmf-wt-meso/prototypes/mesoscopic}
OUT=${MESO_OUT:-/scratch/user/uqaste15/chisep-meso/calib_v2}
export OMP_NUM_THREADS=1
mapfile -t MIX < "$MESO_CALIB_LIST"
i=$SLURM_ARRAY_TASK_ID
NS=${MESO_NSEED:-3}
$PY $HERE/calibrate_iron.py --pops "${MIX[$((i / NS))]}" --seed $((i % NS)) --walkers ${MESO_WALKERS:-6000} --chi ${MESO_CHI:-0.1} --out $OUT
