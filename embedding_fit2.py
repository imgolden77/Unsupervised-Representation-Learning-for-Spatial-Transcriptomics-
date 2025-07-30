import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding_fit3 import CellEmbeddingPipeline, CellEmbeddingDefaultModelConfig, CellEmbeddingDefaultPipelineConfig, CellEmbeddingWandbConfig
import scanpy as sc
import matplotlib.pyplot as plt

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
DATA = 'gse155468'
set_seed(42)

data = ad.read_h5ad('./data/gse155468_preprocessed.h5ad')
data.obs_names_make_unique()
data.obs['batch'] = data.obs['orig.ident'] 

train_num = data.shape[0]
data.obs['split'] = 'train' #즉, 일단은 모든 데이터를 "test"로 표시
tr = np.random.permutation(train_num) #torch.randperm(train_num).numpy()
data.obs['split'][tr[int(train_num*0.8):int(train_num*0.9)]] = 'valid'
data.obs['split'][tr[int(train_num*0.9):]] = 'test'

# CellEmbeddingDefaultModelConfig = {
#     'head_type': 'embedder',
#     'mask_node_rate': 0.75,
#     'mask_feature_rate': 0.25,
#     'max_batch_size': 70000,
# }

# CellEmbeddingDefaultPipelineConfig = {
#     'es': 20,
#     'lr': 5e-4,
#     'wd': 1e-6,
#     'scheduler': 'plat',
#     'epochs': 100,
#     'max_eval_batch_size': 100000,
#     'patience': 5,
#     'workers': 0,
# }

# CellEmbeddingWandbConfig = {
#     "mode":"offline",  # 인터넷 없이 로깅
#     "entity": "juha95-university-of-manchester",  # 엔티티(팀) 이름
#     "project": "CellEmbedding",  # 프로젝트 이름
#     "config": {  # 하이퍼파라미터 정보
#         **CellEmbeddingDefaultModelConfig,
#         **CellEmbeddingDefaultPipelineConfig
#     },
# }

pipeline_config = CellEmbeddingDefaultPipelineConfig.copy()
model_config = CellEmbeddingDefaultModelConfig.copy()
wandb_config =CellEmbeddingWandbConfig.copy()
print(pipeline_config)
print(model_config)

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='./ckpt')
print(pipeline.model)

pipeline.fit(data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            wandb_config= wandb_config,
            label_fields=['celltype'],
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            device = DEVICE
            )

embedding = pipeline.predict(data, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

score_result =pipeline.score(data, # An AnnData object
               label_fields=['celltype'],
               evaluation_config = {
                   'method': 'scanpy', # change to 'scanpy' if 'rapids_singlecell' is not installed; the final scores may vary due to the implementation
                   'batch_size': 50000, # Specify batch size to limit gpu memory usage
               },
               device=DEVICE) # Specify a gpu or cpu for model inference
print(score_result)

data.obsm['emb'] = embedding.cpu().numpy()
sc.pp.neighbors(data, use_rep='emb') 
sc.tl.umap(data) 
plt.rcParams['figure.figsize'] = (12, 12)
ari = score_result['ari']
nmi = score_result['nmi']
sc.pl.umap(data, color='celltype', palette='Paired', title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", size=50 )
plt.savefig(f"./figure/{DATA}_embedding_fit_supCon.png", dpi=300)
plt.show()