#!/bin/bash
#SBATCH --job-name=meso-calib
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-29
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/uqaste15/chisep-meso/slurm_logs/calib_%A_%a.log
set -euo pipefail
PY=/home/uqaste15/miniconda3/envs/qsmxt/bin/python3
HERE=${MESO_DIR:-/scratch/user/uqaste15/src/qsmf-wt-meso/prototypes/mesoscopic}
export OMP_NUM_THREADS=1
RS=(0.5 1 2.5 5 10); XS=(1 3 10); BS=(3 7)
i=$SLURM_ARRAY_TASK_ID
R=${RS[$((i % 5))]}; X=${XS[$(((i / 5) % 3))]}; B=${BS[$((i / 15))]}
$PY $HERE/calibrate_iron.py --radius $R --dchi $X --B0 $B --out /scratch/user/uqaste15/chisep-meso/calib
