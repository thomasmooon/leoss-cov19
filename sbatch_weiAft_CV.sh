#!/bin/bash
#SBATCH --job-name=weiAftCv
#SBATCH --time=12:00:00
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

cd /home/tlinden/leoss
python src/models/weiAft_CV.py

