#!/bin/bash
#SBATCH --job-name=deepsurv_nCV
#SBATCH --array=0-4
#SBATCH --time=16:00:00
#SBATCH -p gpu -n 1 --gres gpu:v100:1
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#
#SBATCH --output=deepsurv.%A_%a.txt
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=thomas.linden@scai.fraunhofer.de

RUNID=77f123013b424613b8a55a2e35ea0b13
module load Anaconda3
source activate /home/tlinden/.conda/envs/leoss/
python src/models/deepsurv_nCV_arrayjob.py --ofold $SLURM_ARRAY_TASK_ID --run_id $RUNID
