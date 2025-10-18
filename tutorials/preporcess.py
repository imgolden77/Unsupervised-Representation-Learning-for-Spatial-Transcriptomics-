import pandas as pd
import anndata as ad
import hdf5plugin
import scanpy as sc
from scipy.sparse import csr_matrix

# 1. Load count matrix and labels
counts = pd.read_csv("./data/GSE87544_Merged_17samples_14437cells_count.txt.gz", sep="\t", index_col=0)
counts = counts.T
labels = pd.read_csv("./data/GSE87544_1443737Cells.SVM.cluster.identity.renamed.csv.gz", index_col=0)
labels.set_index('Cell_ID_temp', inplace=True)

# 2. Match cell ID
common_cells = counts.index.intersection(labels.index)
counts = counts.loc[common_cells]
labels = labels.loc[common_cells]

# 3. Generate AnnData
adata = sc.AnnData(X=counts)
adata.obs = labels
adata.var_names_make_unique()

# print("adata:", adata)
# print("adata.obs:", adata.obs)
# print("adata.var:", adata.var)
# print(adata.obs.columns)

DATA='GSE87544_mouse_brain' #'GSE97930_FrontalCortex_ensg' #'GSE87544_mouse_brain'
adata=ad.read_h5ad(f"./data/{DATA}.h5ad")
print(adata.obs['SVM_clusterID'].unique())
print(adata.X)
adata.X = csr_matrix(adata.X)
print(adata.X)
adata.X.sort_indices()
print(adata.X)

adata.write("./data/GSE87544_mouse_brain.h5ad")