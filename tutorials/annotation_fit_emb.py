import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(".."))

import hdf5plugin
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.utils.ckpt import save_finetuned_model
from CellPLM.pipeline.cell_type_annotation import CellTypeAnnotationPipeline, CellTypeAnnotationDefaultPipelineConfig, CellTypeAnnotationDefaultModelConfig, CellTypeAnnotationWandbConfig
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda:0'
EPOCH = 2000
seed_num =11
set_seed(seed_num)
change ='0826_e2000'

#========DLPFC ALL data download=========
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
# train_adata = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(12)])
# train_adata.obs['batch'] = train_adata.obs['sample_id']  # 핵심 한 줄!

# #========DLPFC sample download=========

# DATA = 'sample_4'

# train_adata = ad.read_h5ad(f'./data/sample/{DATA}.h5ad')

# #========DLPFC data preprocess=========

# train_adata.obs['x_FOV_px'] = train_adata.obs['x']
# train_adata.obs['y_FOV_px'] = train_adata.obs['y']
# train_adata.obs['platform'] = 'visium'
# train_adata.obs['layer'] = train_adata.obs['layer'].cat.add_categories(['Unknown'])
# train_adata.obs['layer'] = train_adata.obs['layer'].fillna('Unknown') 
# train_adata.obs['celltype'] = train_adata.obs['layer'] #DLPFC

# #========MERFISH sample download=========
DATA='MERFISH_0.04_ensg'#'MERFISH_0.04'
train_adata = ad.read_h5ad(f'./data/{DATA}.h5ad')
train_adata.obs_names_make_unique()
train_adata.obs['celltype'] = train_adata.obs['cell_class']
# #=======MERFISH=======
# DATA = 'MERFISH_mouseBrain'
# sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
#                        './data/MERFISH_0.09_ensg.h5ad',
#                        './data/MERFISH_0.14_ensg.h5ad',
#                        './data/MERFISH_0.19_ensg.h5ad',
#                        './data/MERFISH_0.24_ensg.h5ad'])
# samples = [sc.read_h5ad(p) for p in sample_paths]

# train_adata = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

# train_adata.obs['batch'] = train_adata.obs['sample_id'] 
# train_adata.obs['celltype'] = train_adata.obs['cell_class'] #MERFISH

#=========Single cell dataset preprocess======
# DATA ='breast'
##'lung_ensg' #'colon' #'breast' #'gse155468_preprocessed' #'GSE151530_Liver_ensg' # 'GSE131907_Lung_ensg'
##'HumanLiverCancerPatient2_filtered_ensg'(ST) #'HumanLungCancerPatient2_filtered_ensg'(ST) #'GSE97930_FrontalCortex_preprocessed' 
##'demo_train'(hPancread) # 'hPancreas_ensg' #'c_data'(MS)

# train_adata = ad.read_h5ad(f'./data/{DATA}.h5ad')
# train_adata.obs_names_make_unique()
# train_adata.obs['batch'] = train_adata.obs['BATCH'] #colon, breast, lung_ensg
# train_adata.obs['celltype'] = train_adata.obs['Type'] #'GSE151530_Liver_ensg' 
# train_adata.obs['celltype'] = train_adata.obs['Cell_type'] #'GSE131907_Lung_ensg'
# train_adata.obs['batch'] = train_adata.obs['str_batch']#'c_data'(MS) 
# train_adata.var = train_adata.var.set_index('ENSEMBL') #colon
# train_adata.var = train_adata.var.set_index('index_column') #'c_data'(MS)

#==========data preprocess===========

train_num = train_adata.shape[0]
train_adata.obs['split'] = 'train' 
tr = np.random.permutation(train_num) #torch.randperm(train_num).numpy()
train_adata.obs['split'][tr[int(train_num*0.8):int(train_num*0.9)]] = 'valid'
train_adata.obs['split'][tr[int(train_num*0.9):]] = 'test'

#==========Pipeline set===========
pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
wandb_config =CellTypeAnnotationWandbConfig.copy()
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config['out_dim'] = train_adata.obs['celltype'].nunique()
print(pipeline_config, model_config)

pipeline = CellTypeAnnotationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='./ckpt')
pipeline.model
print(pipeline.model)
#==========Fine tuning===========
pipeline.fit(adata= train_adata, # An AnnData object
            train_config= pipeline_config, # The config dictionary we created previously, optional
            wandb_config= wandb_config,
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            label_fields = ['celltype']) # Specify a column in .obs that contains cell type labels

#==========Predict===========
prediction= pipeline.predict(
                train_adata, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
            )
print(prediction.keys())
print(prediction['pred'])

#==========Test===========
score_result= pipeline.score(train_adata, # An AnnData object
                pipeline_config, # The config dictionary we created previously, optional
                split_field = 'split', # Specify a column in .obs to specify train and valid split, optional
                target_split = 'test', # Specify a target split to predict, optional
                label_fields = ['celltype'])  # Specify a column in .obs that contains cell type labels
print(score_result)

train_adata.obsm['emb'] = prediction['latent'].cpu().numpy()
sc.pp.neighbors(train_adata, use_rep='emb', random_state=seed_num) 
sc.tl.umap(train_adata, random_state=seed_num) 
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

best_ari = -1
best_nmi = -1
for res in range(1, 15, 1):
    res = res / 10
    sc.tl.leiden(train_adata, resolution=res, key_added='leiden', random_state=seed_num)
    ari_score = adjusted_rand_score(train_adata.obs['leiden'].to_numpy(), train_adata.obs[['celltype'][0]].to_numpy())
    if ari_score > best_ari:
        best_ari = ari_score
    nmi_score = normalized_mutual_info_score(train_adata.obs['leiden'].to_numpy(), train_adata.obs[['celltype'][0]].to_numpy())
    if nmi_score > best_nmi:
        best_nmi = nmi_score

ari = best_ari
nmi = best_nmi
print(f"{DATA}\nARI={ari}, NMI={nmi}")
# sc.pl.umap(train_adata, color='celltype', palette='Paired', title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", size=50 )
sc.pl.umap(train_adata, 
           color='celltype', 
           palette='Paired', 
           title=f"{DATA}\nARI={ari:.3f}, NMI={nmi:.3f}", 
           size=30,
           legend_fontsize=10,      
            show=False              
            )

filename_prefix=f"{DATA}_annotation{seed_num}_{change}"

save_finetuned_model(
    model=pipeline.model,
    config=model_config,
    save_dir="./ckpt",
    filename_prefix=filename_prefix,
    seed=seed_num,
    adata=train_adata,
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

plt.savefig(f"./figure/{DATA}_annotation_fit_seed{seed_num}_{change}.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Spatial ---

sc.pl.embedding(train_adata, 
                basis="spatial", 
                color="leiden", 
                size=50, 
                show=False, 
                title=f"{DATA}\nARI={ari:.3f}" )

plt.savefig(f"./figure/{DATA}_annotation_SPATIAL_seed{seed_num}_{change}.png", 
            dpi=300, bbox_inches="tight")
plt.close()