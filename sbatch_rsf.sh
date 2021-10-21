#!/bin/bash
#SBATCH --job-name=rsf_nCV
#SBATCH --time=48:00:00
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

RUNID=58576838ef5a4b7481d70c9c5ee4e7fb
cd /home/tlinden/leoss
python src/models/rsf_death_nCV.py
