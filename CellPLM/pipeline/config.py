CellTypeAnnotationWandbConfig = {
    "mode":"offline",  # 인터넷 없이 로깅
    "entity": "juha95-university-of-manchester",  # 엔티티(팀) 이름
    "project": "test",  # 프로젝트 이름
    "config": {  # 하이퍼파라미터 정보
        'drop_node_rate': 0.3,
        'dec_layers': 1,
        'model_dropout': 0.2,
        'mask_node_rate': 0.75,
        'mask_feature_rate': 0.25,
        'dec_mod': 'mlp',
        'latent_mod': 'ae',
        'head_type': 'annotation',
        'max_batch_size': 70000,
        'es': 25,
        'lr': 5e-3,
        'wd': 1e-7,
        'scheduler': 'plat',
        'epochs': 100,
        'max_eval_batch_size': 100000,
        'hvg': 3000,
        'patience': 25,
        'workers': 0,
        },
}

CellTypeAnnotationDefaultModelConfig = {
    'drop_node_rate': 0.3,
    'dec_layers': 1,
    'model_dropout': 0.2,
    'mask_node_rate': 0.75,
    'mask_feature_rate': 0.25,
    'dec_mod': 'mlp',
    'latent_mod': 'gmvae',
    'head_type': 'annotation',
    'max_batch_size': 70000,
}

CellTypeAnnotationDefaultPipelineConfig = {
    'es': 25,
    'lr': 5e-3,
    'wd': 1e-7,
    'scheduler': 'plat',
    'epochs': 100,
    'max_eval_batch_size': 100000,
    'hvg': 3000,
    'patience': 25,
    'workers': 0,
}