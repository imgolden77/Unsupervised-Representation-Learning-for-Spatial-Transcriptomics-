#!/bin/bash

#SBATCH --job-name=train_gpu
#SBATCH --partition=boost_usr_prod            

#SBATCH --account=euhpc_b24_014           
#SBATCH --time=24:00:00                  
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      
#SBATCH --cpus-per-task=1               
#SBATCH --mem=128G                         
#SBATCH --output=logs/cell_annotation/annotation_fit_emb/annotation_emb_%j_0826MERFISH1.out            
#SBATCH --error=logs/cell_annotation/annotation_fit_emb/annotation_emb_%j_0826MERFISH1.err            

#DLPFC #_samples

# load modules
module load python/3.10.8--gcc--8.5.0
module load cuda/12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0

# activate virtual environment
source ~/myenv/bin/activate

# set working directory
cd $WORK/CellPLM  

# check GPU status
nvidia-smi

# Python script execution
python annotation_fit_emb.py
