#!/bin/bash

#SBATCH --job-name=train_gpu
#SBATCH --partition=boost_usr_prod         # GPU 노드 파티션 (Leonardo 기준) boost_usr_prod  

#SBATCH --account=euhpc_b24_014           # ⚠️ 너의 프로젝트 계정
#SBATCH --time=06:00:00                   # 최대 실행 시간
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                      # GPU 1개 사용
#SBATCH --cpus-per-task=1               # CPU 4개
#SBATCH --mem=32G                         # 메모리
#SBATCH --output=imputation_fit_%j_20250717DLPFC_ref_sc_ep100.out            # 표준 출력 로그 (%j = job ID)
#SBATCH --error=imputation_fit_%j_20250717DLPFC_ref_sc_ep100.err             # 표준 에러 로그

# ✅ 모듈 로드
module load python/3.10.8--gcc--8.5.0
module load cuda/12.2
module load cudnn/8.9.7.29-12--gcc--12.2.0

# ✅ 가상환경 활성화
source ~/myenv/bin/activate

# ✅ 작업 디렉토리 이동
cd $WORK/CellPLM  # train.py가 있는 실제 경로로 수정해!

# ✅ GPU 확인 (선택사항)
nvidia-smi

export WANDB_API_KEY=3df0032bb8f54d5c5dceb48e04f50de85f1e72a6

# ✅ Python 코드 실행
python imputation_fit.py
