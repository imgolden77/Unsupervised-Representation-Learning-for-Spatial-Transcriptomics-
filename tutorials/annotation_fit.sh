#!/bin/bash

#SBATCH --job-name=train_gpu
#SBATCH --partition=boost_usr_prod            

#SBATCH --account=euhpc_b24_014           
#SBATCH --time=06:00:00                   
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                     
#SBATCH --cpus-per-task=1               
#SBATCH --mem=32G                         
#SBATCH --output=logs/cell_annotation/annotation_%j_20250717DLPFC_pe_gmvae.out            
#SBATCH --error=logs/cell_annotation/annotation_%j_20250717DLPFC_pe_gmvae.err             

module load python/3.10.8--gcc--8.5.0
module load cuda/12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0

source ~/myenv/bin/activate

cd $WORK/CellPLM  # train.py가 있는 실제 경로로 수정해!

nvidia-smi

python annotation_fit.py
