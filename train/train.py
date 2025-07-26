import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(".."))

import hdf5plugin
import numpy as np
import anndata as ad
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.utils.ckpt import save_finetuned_model
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig, CellTypeAnnotationWandbConfig

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'

sample_paths = sorted(['./data/sample/sample_1.h5ad',
                       './data/sample/sample_2.h5ad',
                       './data/sample/sample_3.h5ad',
                       './data/sample/sample_4.h5ad',
                       './data/sample/sample_5.h5ad',
                       './data/sample/sample_6.h5ad',
                       './data/sample/sample_7.h5ad',
                       './data/sample/sample_8.h5ad',
                       './data/sample/sample_9.h5ad',
                       './data/sample/sample_10.h5ad',
                       './data/sample/sample_11.h5ad',
                       './data/sample/sample_12.h5ad'])
samples = [sc.read_h5ad(p) for p in sample_paths]

# 2. Split by sample
train_samples = samples[:10]
test_samples = samples[10:]

# 3. Concat samples with labels
train_adata = ad.concat(train_samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(10)])
test_adata = ad.concat(test_samples, join='outer', label='sample_id', keys=[f'sample{i+11}' for i in range(2)])

# 4. Assign split labels
test_adata.obs['split'] = 'test'

# 5. Layer: fill NA with "Unknown" for all
for adata in [train_adata, test_adata]:
    adata.obs['batch'] = adata.obs['sample_id']  # 핵심 한 줄!
    adata.obs['layer'] = adata.obs['layer'].cat.add_categories(['Unknown'])
    adata.obs['layer'] = adata.obs['layer'].fillna('Unknown')

train_adata.obs['split'] = 'unknown'

train_idx = []
valid_idx = []

for sample in train_adata.obs['batch'].unique():
    sample_idx = train_adata.obs[train_adata.obs['batch'] == sample].index.to_list()
    train_i, valid_i = train_test_split(sample_idx, test_size=0.1, random_state=42)
    
    train_adata.obs.loc[train_i, 'split'] = 'train'
    train_adata.obs.loc[valid_i, 'split'] = 'valid'

pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
wandb_config =CellTypeAnnotationWandbConfig.copy()
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = train_adata.obs['layer'].nunique()
print(pipeline_config, model_config)

pipeline = CellTypeAnnotationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
pipeline.model
print(pipeline.model)

pipeline.fit(adata= train_adata, # An AnnData object
            train_config= pipeline_config, # The config dictionary we created previously, optional
            wandb_config= wandb_config,
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            label_fields = ['layer']) # Specify a column in .obs that contains cell type labels

save_finetuned_model(
    model= pipeline.model,
    config=model_config,                     # 네가 사용한 overwrite_config
    save_dir='./ckpt',
    filename_prefix='20250712brain_celltypeanno'
)

prediction= pipeline.predict(
                test_adata, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
            )
print(prediction)
score_result= pipeline.score(test_adata, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', # Specify a column in .obs to specify train and valid split, optional
                target_split = 'test', # Specify a target split to predict, optional
                label_fields = ['layer'])  # Specify a column in .obs that contains cell type labels
print(score_result)