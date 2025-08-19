import pandas as pd
import anndata as ad
import hdf5plugin
import scanpy as sc
from scipy.sparse import csr_matrix

# 1. 데이터 불러오기
# counts = pd.read_csv("./data/GSE87544_Merged_17samples_14437cells_count.txt.gz", sep="\t", index_col=0)

# counts = counts.T

# labels = pd.read_csv("./data/GSE87544_1443737Cells.SVM.cluster.identity.renamed.csv.gz", index_col=0)
# # labels.set_index('Cell_ID_temp', inplace=True)


# # print(labels.columns)
# # print(labels.head())

# # 2. cell ID 일치시키기
# common_cells = counts.index.intersection(labels.index)
# counts = counts.loc[common_cells]
# labels = labels.loc[common_cells]

# # 3. AnnData 생성
# adata = sc.AnnData(X=counts)
# adata.obs = labels
# adata.var_names_make_unique()

# print("adata:", adata)
# print("adata.obs:", adata.obs)
# print("adata.var:", adata.var)

# print(adata.obs.columns)
DATA='GSE87544_mouse_brain' #'GSE97930_FrontalCortex_ensg' #'GSE87544_mouse_brain'
adata=ad.read_h5ad(f"./data/{DATA}.h5ad")
# print(adata.obs['SVM_clusterID'].unique())
# print(adata.X)
# adata.X = csr_matrix(adata.X)
# print(adata.X)
# adata.X.sort_indices()
# print(adata.X)

# adata.write("./data/GSE87544_mouse_brain.h5ad")

print(adata)
print(adata.obs)
print(adata.var)

# #=============DLPFC data preprocess============
# # 1. Load all 12 samples
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
#                        './data/sample/sample_12.h5ad'])  # 12개 h5ad 경로 리스트
# samples = [ad.read_h5ad(p) for p in sample_paths]

# # 3. Train 샘플 concat
# query_data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(12)])
# query_data.obs['layer'] = query_data.obs['layer'].cat.add_categories(['Unknown'])
# query_data.obs['layer'] = query_data.obs['layer'].fillna('Unknown')
# # query_data.obs['batch']=query_data.obs['sample_id']
# # query_data.obs['platform'] = 'cosmx'
# # query_data.obs['x_FOV_px'] = query_data.obs['x']
# # query_data.obs['y_FOV_px'] = query_data.obs['y']
# ref_data = ad.read_h5ad(f'./data/GSE97930_FrontalCortex_ensg.h5ad')
# ref_data.X = csr_matrix(ref_data.X)



#=======MERFISH=======
# sample_paths = sorted(['./data/MERFISH_0.04_ensg.h5ad',
#                        './data/MERFISH_0.09_ensg.h5ad',
#                        './data/MERFISH_0.14_ensg.h5ad',
#                        './data/MERFISH_0.19_ensg.h5ad',
#                        './data/MERFISH_0.24_ensg.h5ad'])
# samples = [ad.read_h5ad(p) for p in sample_paths]

# query_data = ad.concat(samples, join='outer', label='sample_id', keys=[f'sample{i+1}' for i in range(5)])

# query_data.obs['batch'] = query_data.obs['sample_id']  # 핵심 한 줄!
# ref_data = ad.read_h5ad(f'./data/GSE87544_mouse_brain.h5ad')
# ref_data.obs['batch'] = ref_data.obs['SVM_clusterID']
# ref_data.X = csr_matrix(ref_data.X)

# print("query_data:", query_data)
# # print("query_data.obs:", query_data.obs)
# # print("query_data.var:",query_data.var)
# # print("query_data.X:",query_data.X)

# print("ref_data:", ref_data)
# print("ref_data.obs:", ref_data.obs)
# print("ref_data.var:",ref_data.var)
# print("ref_data.X:",ref_data.X)