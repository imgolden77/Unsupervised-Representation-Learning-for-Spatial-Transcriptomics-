import warnings
warnings.filterwarnings("ignore")
import json
import sys
import os
sys.path.append(os.path.abspath(".."))

import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.utils.data import stratified_sample_genes_by_sparsity
from CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig, ImputationWandbConfig

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
set_seed(11)
# DATASET ='Liver'

# 1. Load all 12 samples
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
                       './data/sample/sample_12.h5ad'])  # 12개 h5ad 경로 리스트
samples = [ad.read_h5ad(p) for p in sample_paths]

# 3. Train 샘플 concat
query_data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(12)])
query_data.obs['layer'] = query_data.obs['layer'].cat.add_categories(['Unknown'])
query_data.obs['layer'] = query_data.obs['layer'].fillna('Unknown')
query_data.obs['batch']=query_data.obs['sample_id']
query_data.obs['platform'] = 'cosmx'
query_data.obs['x_FOV_px'] = query_data.obs['x']
query_data.obs['y_FOV_px'] = query_data.obs['y']
ref_data = ad.read_h5ad(f'./data/GSE97930_FrontalCortex_preprocessed.h5ad')
ref_data.X = csr_matrix(ref_data.X)

pipeline_config = ImputationDefaultPipelineConfig.copy()
model_config = ImputationDefaultModelConfig.copy()
wandb_config =ImputationWandbConfig.copy()
print(pipeline_config)
print(model_config)

pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
print(pipeline.model)

preprocessed = pipeline.common_preprocess(query_data, hvg=0, covariate_fields=None, ensembl_auto_conversion=True)

target_genes = stratified_sample_genes_by_sparsity(preprocessed, seed=11)
query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
query_data[:, target_genes].X = 0

train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

train_data.obs['split'] = 'train'
train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid'
train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid'

query_genes = [g for g in query_data.var.index if g not in target_genes]
query_batches = list(query_data.obs['batch'].unique())
ref_batches = list(ref_data.obs['batch'].unique())
batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
    [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))


pipeline.fit(train_data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            wandb_config= wandb_config,
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
            device = DEVICE,
            )

prediction =pipeline.predict(
        query_data, # An AnnData object
        pipeline_config, # The config dictionary we created previously, optional
        device = DEVICE,
    )
print(prediction)

score_result=pipeline.score(
                query_data, # An AnnData object
                evaluation_config = {'target_genes': target_genes}, # The config dictionary we created previously, optional
                label_fields = ['truth'], # A field in .obsm that stores the ground-truth for evaluation
                device = DEVICE,
)
print(score_result)