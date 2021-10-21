#!/bin/bash
#SBATCH --job-name=weiAft
#SBATCH --array=0-4
#SBATCH --time=12:00:00 # hh:mm:ss
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

RUNID=1b7556d626884f9187b938ca68ad747e
cd /home/tlinden/leoss
python src/models/weibull_aft_nCV.py --ofold $SLURM_ARRAY_TASK_ID --run_id $RUNID

