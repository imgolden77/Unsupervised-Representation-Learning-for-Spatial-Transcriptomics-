import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))

# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad
import scanpy as sc
import numpy as np
# from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from CellPLM.pipeline.experimental import symbol_to_ensembl

DATA='GSE87544_mouse_brain_ensg'  #'lung' #'colon' #'breast' #'gse155468', 'c_data'(MS)

# 1. Load all 12 samples
data = ad.read_h5ad(f'./data/{DATA}.h5ad')

# data.var.index = symbol_to_ensembl(data.var.index.tolist())
# data.var_names_make_unique()

print(data.var)

# 전처리 끝난 AnnData를 저장
# data.write_h5ad("./data/GSE97930_FrontalCortex_ensg.h5ad")