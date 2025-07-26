import torch
import json
import os

def save_finetuned_model(model, config: dict, save_dir: str, filename_prefix: str):
    """
    Fine-tuned 모델을 CellPLM 형식에 맞게 저장합니다.

    Parameters:
    - model: 학습된 OmicsFormer 모델
    - config: overwrite_config 또는 모델 config (dict)
    - save_dir: 저장할 디렉토리 (예: './ckpt/')
    - filename_prefix: 저장 파일 접두어 (예: 'fine_tuned_sample9')
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 가중치 저장 (model_state_dict 포함)
    ckpt_path = os.path.join(save_dir, f'{filename_prefix}.best.ckpt')
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

    # 2. config 저장
    config_path = os.path.join(save_dir, f'{filename_prefix}.config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'✅ Fine-tuned checkpoint saved to:\n- {ckpt_path}\n- {config_path}')