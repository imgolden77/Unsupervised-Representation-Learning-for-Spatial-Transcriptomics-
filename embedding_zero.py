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
DATA = 'MERFISH_mouse_brain' #'lung_ensg' #'colon' #'breast' #'gse155468' #'GSE151530_Liver_ensg' # 'GSE131907_Lung_ensg'
#'HumanLiverCancerPatient2_filtered_ensg'(ST) #'HumanLungCancerPatient2_filtered_ensg'(ST) #'GSE97930_FrontalCortex_preprocessed' 
#'demo_train'(hPancread) # 'hPancreas_ensg' #'c_data'(MS)
set_seed(42)

#========Sing dataset preprocess==========
# data = ad.read_h5ad(f'./data/{DATA}.h5ad')
# data.obs_names_make_unique()
# data.obs['celltype'] = data.obs['Cell_type'] #'GSE131907_Lung_ensg'
# data.obs['celltype'] = data.obs['Type'] #'GSE151530_Liver_ensg' 
# data.obs['celltype'] = data.obs['Celltype'] #'hPancreas_ensg'
# data.obs['batch'] = data.obs['BATCH'] #colon, breast, lung
# data.obs['batch'] = data.obs['str_batch']#'c_data'(MS) 
# data.var = data.var.set_index('ENSEMBL') #colon
# data.var = data.var.set_index('index_column') #'c_data'(MS)


#========DLPFC========
# sample_paths = sorted(['./data/sample/sample_1.h5ad',
#                        './data/sample/sample_2.h5ad',
#                        './data/sample/sample_3.h5ad',
#                        './data/sample/sample_4.h5ad',
#                        './data/sample/sample_5.h5ad',
#                        './data/sample/sample_6.h5ad',
#                        './data/sample/sample_7.h5ad',
#                        './data/sample/sample_8.h5ad',
#                        './data/sample/sample_9.h5ad',
#                        './data/sample/sample_10.h5ad',
#                        './data/sample/sample_11.h5ad',
#                        './data/sample/sample_12.h5ad'])
# samples = [sc.read_h5ad(p) for p in sample_paths]

# data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(12)])

# data.obs['batch'] = data.obs['sample_id']  # 핵심 한 줄!
# data.obs['x_FOV_px'] = data.obs['x']
# data.obs['y_FOV_px'] = data.obs['y']
# data.obs['platform'] = 'visium'
# data.obs['layer'] = data.obs['layer'].cat.add_categories(['Unknown'])
# data.obs['layer'] = data.obs['layer'].fillna('Unknown')


#=======MERFISH=======
sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
                       './data/MERFISH_0.09_ensg.h5ad',
                       './data/MERFISH_0.14_ensg.h5ad',
                       './data/MERFISH_0.19_ensg.h5ad',
                       './data/MERFISH_0.24_ensg.h5ad'])
samples = [sc.read_h5ad(p) for p in sample_paths]

data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

data.obs['batch'] = data.obs['sample_id']  # 핵심 한 줄!
# data.obs['layer'] = data.obs['layer'].cat.add_categories(['Unknown'])
# data.obs['layer'] = data.obs['layer'].fillna('Unknown')


#======cell embedding zeroshot =====

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='./ckpt')
pipeline.model

embedding = pipeline.predict(data, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

score_result =pipeline.score(data, # An AnnData object
               label_fields=['cell_class'],
               evaluation_config = {
                   'method': 'scanpy', # change to 'scanpy' if 'rapids_singlecell' is not installed; the final scores may vary due to the implementation
                   'batch_size': 50000, # Specify batch size to limit gpu memory usage
               },
               device=DEVICE) # Specify a gpu or cpu for model inference
print(score_result)

data.obsm['emb'] = embedding.cpu().numpy()
sc.pp.neighbors(data, use_rep='emb') 

sc.tl.umap(data)
# plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams.update({
    'figure.figsize': (15, 15),
    'axes.titlesize': 24,        # 타이틀 크기
    'axes.labelsize': 18,        # x/y축 라벨 크기
    'legend.fontsize': 18,       # 범례 글씨 크기
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})
ari = score_result['ari']
nmi = score_result['nmi']
sc.pl.umap(
            data, 
           color='cell_class', 
           palette='Paired', 
           title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", 
           size=50,
           legend_fontsize=20,      # 범례 글씨 크게
            legend_loc='on data',  # 또는 'bottom'으로 바꿔도 가능
            show=False              # 그림 그리지 않고 matplotlib figure만 반환
            )
plt.savefig(f"./figure/{DATA}_embedding_zero.png", dpi=300, bbox_inches='tight')
plt.show()