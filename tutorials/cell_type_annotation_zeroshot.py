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
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationZeroShotPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig

DATASET = 'sample1' # 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7', 'sample8', 'sample9', 'sample10', 'sample11', 'sample12'
PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'

set_seed(42)
if DATASET == 'sample1':
    data = ad.read_h5ad(f'./data/sample/sample_1.h5ad')
elif DATASET == 'sample2':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_2.h5ad')
elif DATASET == 'sample3':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_3.h5ad')
elif DATASET == 'sample4':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_4.h5ad')
elif DATASET == 'sample5':
    data= ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_5.h5ad')
elif DATASET == 'sample6':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_6.h5ad')
elif DATASET == 'sample7':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_7.h5ad')
elif DATASET == 'sample8':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_8.h5ad')
elif DATASET == 'sample9':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_9.h5ad')
elif DATASET == 'sample10':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_10.h5ad')
elif DATASET == 'sample11':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_11.h5ad')
elif DATASET == 'sample12':
    data = ad.read_h5ad(f'/content/drive/MyDrive/CellPLM-main/data/sample/sample_12.h5ad')

train_num = data.shape[0]
data.obs['layer'] = data.obs['layer'].cat.add_categories(['Unknown'])
data.obs['layer'] = data.obs['layer'].fillna('Unknown')
data.var_names_make_unique()

data.obs['split'] = 'train' 
tr = np.random.permutation(train_num) #torch.randperm(train_num).numpy()
data.obs['split'][tr[int(train_num*0.8):int(train_num*0.9)]] = 'valid'
data.obs['split'][tr[int(train_num*0.9):]] = 'test'

pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = data.obs['layer'].nunique()

pipeline = CellTypeAnnotationZeroShotPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
pipeline.model

pipeline.predict(
                data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
            )

pipeline.score(data, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', # Specify a column in .obs to specify train and valid split, optional
                target_split = 'test', # Specify a target split to predict, optional
                label_fields = ['layer'])  # Specify a column in .obs that contains cell type labels