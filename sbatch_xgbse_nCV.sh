#!/bin/bash
#SBATCH --job-name=xgbse_nCV
#SBATCH --array=0-4
#SBATCH --time=24:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#
#SBATCH --output=%x.%A_%a.txt
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=thomas.linden@scai.fraunhofer.de

module load Anaconda3
source activate /home/tlinden/.conda/envs/leoss/

RUNID=e74bf4d3ae5b4db79e007fbe6a6a8102
cd /home/tlinden/leoss
python src/models/xgbse_sWeiAft_nCV.py --ofold $SLURM_ARRAY_TASK_ID --run_id $RUNID

