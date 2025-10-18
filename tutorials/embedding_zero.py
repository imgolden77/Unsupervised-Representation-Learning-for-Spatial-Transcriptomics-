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

seed_num=123
set_seed(seed_num)

#========Sing dataset preprocess==========
# DATA = 'MERFISH_mouse_brain' #'lung_ensg' #'colon' #'breast' #'gse155468' #'GSE151530_Liver_ensg' # 'GSE131907_Lung_ensg'
#'HumanLiverCancerPatient2_filtered_ensg'(ST) #'HumanLungCancerPatient2_filtered_ensg'(ST) #'GSE97930_FrontalCortex_preprocessed' 
#'demo_train'(hPancread) # 'hPancreas_ensg' #'c_data'(MS)
# data = ad.read_h5ad(f'./data/{DATA}.h5ad')
# data.obs_names_make_unique()
# data.obs['celltype'] = data.obs['Cell_type'] #'GSE131907_Lung_ensg'
# data.obs['celltype'] = data.obs['Type'] #'GSE151530_Liver_ensg' 
# data.obs['celltype'] = data.obs['Celltype'] #'hPancreas_ensg'
# data.obs['batch'] = data.obs['BATCH'] #colon, breast, lung
# data.obs['batch'] = data.obs['str_batch']#'c_data'(MS) 
# data.var = data.var.set_index('ENSEMBL') #colon
# data.var = data.var.set_index('index_column') #'c_data'(MS)

# # #========DLPFC sample download=========
# DATA = 'sample_4'
# data = ad.read_h5ad(f'./data/sample/{DATA}.h5ad')
#========DLPFC========
# DATA = 'DLPFC'
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
# data.obs['celltype'] = data.obs['layer'] #DLPFC

# #========MERFISH sample download=========
DATA='MERFISH_0.04_ensg'#'MERFISH_0.04'
data = ad.read_h5ad(f'./data/{DATA}.h5ad')
data.obs_names_make_unique()
data.obs['celltype'] = data.obs['cell_class']

#=======MERFISH=======
# DATA = 'MERFISH_mouseBrain'
# sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
#                        './data/MERFISH_0.09_ensg.h5ad',
#                        './data/MERFISH_0.14_ensg.h5ad',
#                        './data/MERFISH_0.19_ensg.h5ad',
#                        './data/MERFISH_0.24_ensg.h5ad'])
# samples = [sc.read_h5ad(p) for p in sample_paths]

# data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

# data.obs['batch'] = data.obs['sample_id']  
# data.obs['celltype'] = data.obs['cell_class'] #MERFISH


#======cell embedding zeroshot =====
pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='./ckpt')
pipeline.model
print(pipeline.model)
embedding = pipeline.predict(data, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

score_result =pipeline.score(data, # An AnnData object
               label_fields=['celltype'],
               evaluation_config = {
                   'method': 'scanpy', # change to 'scanpy' if 'rapids_singlecell' is not installed; the final scores may vary due to the implementation
                   'batch_size': 50000, # Specify batch size to limit gpu memory usage
                   'random_state': seed_num
               },
               device=DEVICE) # Specify a gpu or cpu for model inference
print(score_result)

data.obsm['emb'] = embedding.cpu().numpy()
sc.pp.neighbors(data, use_rep='emb', random_state=seed_num) 

sc.tl.umap(data, random_state=seed_num)
# plt.rcParams['figure.figsize'] = (15, 15)
# plt.rcParams.update({
#     'figure.figsize': (15, 15),
#     'axes.titlesize': 24,        
#     'axes.labelsize': 18,        
#     'legend.fontsize': 18,      
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14
# })

# ====DLPFC single====
plt.rcParams.update({
    'figure.figsize': (5, 5),
    'axes.titlesize': 15,       
    'axes.labelsize': 12,        
    'legend.fontsize':10,      
})
ari = score_result['ari']
nmi = score_result['nmi']
sc.pl.umap(
            data, 
           color='celltype', 
           palette='Paired', 
           title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", 
           size=30,     
            show=False             
            )

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()

plt.legend(
    handles,
    labels,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),  
    fontsize=18,
    ncol=1,                     
    frameon=False
)

plt.savefig(f"./figure/{DATA}_embedding_zero_{seed_num}_0826.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Spatial ---
sc.tl.leiden(data, key_added="leiden_cellplm", resolution=score_result['best_resolution'], random_state=seed_num)
sc.pl.embedding(data, 
                basis="spatial", 
                color="leiden_cellplm", 
                size=50, 
                show=False, 
                title=f"{DATA}\nARI={ari:.3f}" )

plt.savefig(f"./figure/{DATA}_embeddingzeroshot_SPATIAL_seed{seed_num}_0826.png", 
            dpi=300, bbox_inches="tight")
plt.close()