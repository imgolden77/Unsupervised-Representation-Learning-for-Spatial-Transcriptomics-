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
from CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
set_seed(42)

#========DLPFC========
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
samples = [ad.read_h5ad(p) for p in sample_paths]

query_data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(12)])
query_data.obs['layer'] = query_data.obs['layer'].cat.add_categories(['Unknown'])
query_data.obs['layer'] = query_data.obs['layer'].fillna('Unknown')
query_data.obs['batch']=query_data.obs['sample_id']
query_data.obs['platform'] = 'cosmx'
query_data.obs['x_FOV_px'] = query_data.obs['x']
query_data.obs['y_FOV_px'] = query_data.obs['y']

#=======MERFISH=======
# sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
#                        './data/MERFISH_0.09_ensg.h5ad',
#                        './data/MERFISH_0.14_ensg.h5ad',
#                        './data/MERFISH_0.19_ensg.h5ad',
#                        './data/MERFISH_0.24_ensg.h5ad'])
# samples = [ad.read_h5ad(p) for p in sample_paths]

# query_data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

# query_data.obs['batch'] = query_data.obs['sample_id']  

# ---- Load pretrained gene list --------------------------------------------
with open(f'./ckpt/{PRETRAIN_VERSION}.config.json') as f:
    cfg = json.load(f)
pretrained_genes = set(cfg['gene_list'])

# ---- Filter the AnnData objects -------------------------------------------
# retain only genes recognized by the pretrained model
keep_genes = [g for g in query_data.var.index if g in pretrained_genes]
diff_genes = [g for g in query_data.var.index if g not in pretrained_genes]
print("삭제된 유전자 수 :", len(diff_genes))
query_data = query_data[:, keep_genes]
target_genes = stratified_sample_genes_by_sparsity(query_data, seed=11) # This is for reproducing the hold-out gene lists in our paper
# remove held‑out genes not present in the pretrained model
query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
query_data[:, target_genes].X = 0

pipeline_config = ImputationDefaultPipelineConfig.copy()
model_config = ImputationDefaultModelConfig.copy()
print(pipeline_config)
print(model_config)

pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
print(pipeline.model)

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