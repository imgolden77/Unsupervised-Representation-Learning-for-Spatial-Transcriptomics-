# import torch
# import json
# import os

# def save_finetuned_model(model, config: dict, save_dir: str, filename_prefix: str):
#     """
#     Fine-tuned 모델을 CellPLM 형식에 맞게 저장합니다.

#     Parameters:
#     - model: 학습된 OmicsFormer 모델
#     - config: overwrite_config 또는 모델 config (dict)
#     - save_dir: 저장할 디렉토리 (예: './ckpt/')
#     - filename_prefix: 저장 파일 접두어 (예: 'fine_tuned_sample9')
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # 1. 가중치 저장 (model_state_dict 포함)
#     ckpt_path = os.path.join(save_dir, f'{filename_prefix}.best.ckpt')
#     torch.save({'model_state_dict': model.state_dict()}, ckpt_path)

#     # 2. config 저장
#     config_path = os.path.join(save_dir, f'{filename_prefix}.config.json')
#     with open(config_path, 'w') as f:
#         json.dump(config, f, indent=4)

#     print(f'✅ Fine-tuned checkpoint saved to:\n- {ckpt_path}\n- {config_path}')

# CellPLM/utils/ckpt.py
import os, sys, json, time, hashlib, platform, subprocess
import torch
import numpy as np

def _env_info():
    import torch.backends.cudnn as cudnn
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(i))
    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "pytorch": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": getattr(cudnn, "version", lambda: None)(),
        "gpu_names": gpus,
        "gpu_count": torch.cuda.device_count(),
        # 재현성 관련 환경변수도 같이 저장
        "env_vars": {
            k: os.environ.get(k)
            for k in [
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            ]
        },
    }

def _git_info():
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        info["commit"], info["dirty"] = commit, dirty
    except Exception:
        pass
    return info

def _weights_hash(model):
    try:
        h = hashlib.sha256()
        for k, v in model.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    except Exception:
        return None

def _adata_summary(adata, split_field, label_field):
    info = {"n_cells": None, "n_genes": None}
    try:
        info["n_cells"] = int(adata.n_obs)
        info["n_genes"] = int(adata.n_vars)
    except Exception:
        return info

    # split 통계 + 인덱스 해시(스플릿이 달라졌는지 감지)
    if split_field and split_field in adata.obs:
        vc = adata.obs[split_field].value_counts().to_dict()
        idx_hash = {}
        for k in sorted(vc.keys()):
            idx = np.where(adata.obs[split_field].to_numpy() == k)[0]
            idx_hash[k] = hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:16]
        info["split"] = {"counts": vc, "index_hash": idx_hash}

    # 라벨 분포
    if label_field and label_field in adata.obs:
        try:
            info["label_counts"] = adata.obs[label_field].value_counts().to_dict()
        except Exception:
            pass
    return info

def save_finetuned_model(
    model,
    config: dict,
    save_dir: str,
    filename_prefix: str,
    *,
    # ↓↓↓ 추가 가능(원치 않으면 안 넘겨도 됨)
    seed: int | None = None,
    adata=None,
    split_field: str | None = None,
    label_field: str | None = None,
    extra_metrics: dict | None = None,   # {'ari': ..., 'nmi': ...} 등
    notes: str | None = None,
    include_git: bool = True,            # Git 기록 끄려면 False
    optimizer=None,                      # 옵티마이저/스케줄러 상태 요약
    scheduler=None,
):
    """
    Fine-tuned 모델과 config(+실험 컨텍스트)를 함께 저장.
    생성:
      - <prefix>.best.ckpt         (가중치)
      - <prefix>.config.json       (하이퍼파라미터 + __context__)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 체크포인트(가중치) 저장
    ckpt_path = os.path.join(save_dir, f"{filename_prefix}.best.ckpt")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # 2) 컨텍스트 구성
    ctx = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "env": _env_info(),
        "model": {
            "class": type(model).__name__,
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "weights_sha256_16": _weights_hash(model),
        },
        "data": _adata_summary(adata, split_field, label_field),
        "metrics": extra_metrics or {},
        "notes": notes or "",
    }
    if include_git:
        ctx["code"] = _git_info()

    # 옵티마이저/스케줄러 요약(상태 전체 저장 아님: 파일이 너무 커지니까 요약만)
    try:
        if optimizer is not None:
            ctx["optimizer"] = {
                "type": type(optimizer).__name__,
                "param_groups": [
                    {k: v for k, v in g.items() if k in {"lr", "weight_decay", "betas", "eps"}}
                    for g in optimizer.param_groups
                ],
            }
    except Exception:
        pass
    try:
        if scheduler is not None:
            ctx["scheduler"] = {"type": type(scheduler).__name__}
    except Exception:
        pass

    # 3) 기존 config에 컨텍스트 합치기
    full_config = dict(config)
    full_config["__context__"] = ctx

    config_path = os.path.join(save_dir, f"{filename_prefix}.config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)

    print(
        "✅ Fine-tuned checkpoint saved to:\n"
        f"- {ckpt_path}\n- {config_path}"
    )
