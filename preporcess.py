import pandas as pd
import anndata as ad
import hdf5plugin
import scanpy as sc

data = ad.read_h5ad(f'./data/GSE97930_FrontalCortex_preprocessed.h5ad')
# sc.pp.calculate_qc_metrics (data, inplace=True)
# st_data = ad.read_h5ad(f'./data/sample/sample_1.h5ad')
# st_data = ad.read_h5ad(f'./data/HumanLungCancerPatient2_filtered_ensg.h5ad')
# sq_data = ad.read_h5ad(f'./data/GSE131907_Lung_ensg.h5ad')

# obs: 필요한 컬럼만 남기고 다 삭제
# data.obs = data.obs[["n_genes_by_counts"]]

# var: 필요한 컬럼만 남기고 다 삭제
# data.var = data.var[["n_cells_by_counts"]]

# data.obs["donor"] = data.obs_names.str.split("_").str[0]  # Ex1
# data.obs["batch"] = data.obs_names.str.split("_").str[1]  # fcx8
# data.obs["barcode"] = data.obs_names.str.split("_").str[-1]
# print(data.obs.nunique())
# print(data.obs["batch"].unique())
# print(data.obs["batch"].value_counts())

# print("data:", data)
# print("sq_data.obs:", data.obs)
# print("sq_data.var:", data.var)
# data.write("./data/GSE97930_FrontalCortex_preprocessed.h5ad")
print(data)
print(data.obs)
print(data.var)