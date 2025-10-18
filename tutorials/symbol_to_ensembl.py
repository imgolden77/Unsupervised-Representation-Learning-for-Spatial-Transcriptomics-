import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))

import anndata as ad
import scanpy as sc
import numpy as np
import random
import matplotlib.pyplot as plt
from CellPLM.pipeline.experimental import symbol_to_ensembl

DATA='GSE87544_mouse_brain_ensg'  #'lung' #'colon' #'breast' #'gse155468', 'c_data'(MS)
data = ad.read_h5ad(f'./data/{DATA}.h5ad')

data.var.index = symbol_to_ensembl(data.var.index.tolist())
data.var_names_make_unique()

data.write_h5ad("./data/GSE97930_FrontalCortex_ensg.h5ad")