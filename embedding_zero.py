import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))
import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import scanpy as sc
import matplotlib.pyplot as plt

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
DATA ='gse155468'
set_seed(42)

data = ad.read_h5ad('./data/gse155468_preprocessed.h5ad')
data.obs_names_make_unique()

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='./ckpt')
pipeline.model

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
sc.pl.umap(data, color='celltype', palette='Paired', title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}")
plt.savefig(f"./figure/{DATA}_embedding_zero2.png", dpi=300)
plt.show()