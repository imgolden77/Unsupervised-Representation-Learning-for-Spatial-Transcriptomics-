import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.CellPLM.pipeline.cell_embedding_fit import CellEmbeddingPipeline, CellEmbeddingDefaultModelConfig, CellEmbeddingDefaultPipelineConfig, CellEmbeddingWandbConfig
import scanpy as sc
import matplotlib.pyplot as plt
from CellPLM.utils.ckpt import save_finetuned_model

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
EPOCH = 300
seed_num =123
set_seed(seed_num)
change ='0826'

# # ========DLPFC ALL data download=========
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

# #========DLPFC sample download=========

# DATA = 'sample_4'
# data = ad.read_h5ad(f'./data/sample/{DATA}.h5ad')

# # # ========DLPFC data preprocess=========

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
# # # ============MERFISH=============
# DATA = 'MERFISH_mouseBrain'
# sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
#                        './data/MERFISH_0.09_ensg.h5ad',
#                        './data/MERFISH_0.14_ensg.h5ad',
#                        './data/MERFISH_0.19_ensg.h5ad',
#                        './data/MERFISH_0.24_ensg.h5ad'])
# samples = [sc.read_h5ad(p) for p in sample_paths]

# data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

# data.obs['batch'] = data.obs['sample_id']  # 핵심 한 줄!
# data.obs['celltype'] = data.obs['cell_class'] #MERFISH

# #========Sing dataset preprocess==========
# DATA = 'gse155468_preprocessed'  #'lung_ensg' #'colon' #'breast' #'gse155468_preprocessed' #'GSE151530_Liver_ensg' # 'GSE131907_Lung_ensg'
# # # #'HumanLiverCancerPatient2_filtered_ensg'(ST) #'HumanLungCancerPatient2_filtered_ensg'(ST) #'GSE97930_FrontalCortex_preprocessed' 
# # # #'demo_train'(hPancread) # 'hPancreas_ensg' #'c_data'(MS)
# data = ad.read_h5ad(f'./data/{DATA}.h5ad')
# data.obs_names_make_unique()
# # data.obs['batch'] = data.obs['BATCH'] #colon, breast, lung_ensg

# data.obs['celltype'] = data.obs['Cell_type'] #'GSE131907_Lung_ensg'
# # data.obs['celltype'] = data.obs['Type'] #'GSE151530_Liver_ensg' 
# # data.var = data.var.set_index('ENSEMBL') #colon


#==========data preprocess===========
train_num = data.shape[0]
data.obs['split'] = 'train' 
tr = np.random.permutation(train_num) #torch.randperm(train_num).numpy()
data.obs['split'][tr[int(train_num*0.8):int(train_num*0.9)]] = 'valid'
data.obs['split'][tr[int(train_num*0.9):]] = 'test'

#==========Pipeline set===========
pipeline_config = CellEmbeddingDefaultPipelineConfig.copy()
model_config = CellEmbeddingDefaultModelConfig.copy()
wandb_config =CellEmbeddingWandbConfig.copy()
print(pipeline_config)
print(model_config)

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='./ckpt')
print(pipeline.model)

#==========Fine tuning===========
pipeline.fit(data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            wandb_config= wandb_config,
            label_fields=['celltype'],
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            device = DEVICE
            )

#==========Predict And Score===========
embedding = pipeline.predict(data, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

score_result =pipeline.score(data, # An AnnData object
               label_fields=['celltype'],
               evaluation_config = {
                   'method': 'scanpy', # change to 'scanpy' if 'rapids_singlecell' is not installed; the final scores may vary due to the implementation
                   'batch_size': 50000, # Specify batch size to limit gpu memory usage
                   'random_state': seed_num,
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
#====DLPFC single====
plt.rcParams.update({
    'figure.figsize': (5, 5),
    'axes.titlesize': 15,        
    'axes.labelsize': 12,        
    'legend.fontsize':10,       
})
ari = score_result['ari']
nmi = score_result['nmi']
sc.pl.umap(data, 
           color='celltype', 
           palette='Paired', 
           title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", 
           size=30,
            legend_fontsize=10,      
            show=False            
            )

filename_prefix=f"0823{DATA}_embedding{seed_num}_{change}"

save_finetuned_model(
    model=pipeline.model,
    config=model_config,
    save_dir="./ckpt",
    filename_prefix=filename_prefix,
    seed=seed_num,
    adata=data,
    split_field="split",
    label_field="celltype",
    extra_metrics={**score_result, "evaluation_config": {"method":"scanpy","random_state": seed_num}},          # {'ari': ..., 'nmi': ...}
    notes=f"DATA={DATA}",
    include_git=True,                   
    optimizer=None,                    
    scheduler=None,
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

plt.savefig(f"./figure/{DATA}_embedding_supCon_seed{seed_num}_{change}.png", dpi=300, bbox_inches='tight') #nr0.75fr0.25
plt.close()

# --- Spatial ---
sc.tl.leiden(data, key_added="leiden_cellplm", resolution=score_result['best_resolution'], random_state=seed_num)

sc.pl.embedding(data, 
                basis="spatial", 
                color="leiden_cellplm", 
                size=50, 
                show=False,
                title=f"{DATA}\nARI={ari:.3f}" )

plt.savefig(f"./figure/{DATA}_SPATIAL_seed{seed_num}_{change}.png", 
            dpi=300, bbox_inches="tight")
plt.close()