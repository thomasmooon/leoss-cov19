#!/bin/bash
#SBATCH --job-name=cox
#SBATCH --array=0-4
#SBATCH --time=6:00:00
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

RUNID=b027c38fe49a4b9ab9b867f00ee32e5a
cd /home/tlinden/leoss
python src/models/cox_nCV.py --ofold $SLURM_ARRAY_TASK_ID --run_id $RUNID



