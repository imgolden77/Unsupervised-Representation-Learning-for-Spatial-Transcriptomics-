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

PRETRAIN_VERSION = '20250712brain_celltypeanno'
DEVICE = 'cuda:0'
#'./data/sample/sample_12.h5ad'
test_adata=ad.read_h5ad(f'./data/sample/sample_11.h5ad')

# 4. Assign split labels
# train_adata.obs['split'] = 'train'
test_adata.obs['split'] = 'test'

test_adata.obs['batch'] = test_adata.obs['sample_id']  # 핵심 한 줄!
test_adata.obs['layer'] = test_adata.obs['layer'].cat.add_categories(['Unknown'])
test_adata.obs['layer'] = test_adata.obs['layer'].fillna('Unknown')

pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
wandb_config =CellTypeAnnotationWandbConfig.copy()
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = test_adata.obs['layer'].nunique()
pipeline_config, model_config

pipeline = CellTypeAnnotationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
pipeline.model

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