#!/bin/bash

#SBATCH --job-name=train_gpu
#SBATCH --partition=boost_usr_prod            

#SBATCH --account=euhpc_b24_014           
#SBATCH --time=06:00:00                   
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      
#SBATCH --cpus-per-task=1              
#SBATCH --mem=32G                         
#SBATCH --output=logs/cell_embedding/zero_shot/embedding_%j_0826DLPFC4_0.out            
#SBATCH --error=logs/cell_embedding/zero_shot/embedding_%j_0826DLPFC4_0.err            

module load python/3.10.8--gcc--8.5.0
module load cuda/12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0

source ~/myenv/bin/activate

cd $WORK/CellPLM  

python embedding_zero.py
