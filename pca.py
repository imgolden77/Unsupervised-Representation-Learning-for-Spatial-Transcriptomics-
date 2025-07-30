import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath(".."))

import hdf5plugin

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad
import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from CellPLM.pipeline.experimental import symbol_to_ensembl


SEED = 42  # 원하는 고정 값
np.random.seed(SEED)
random.seed(SEED)
DATA='GSE151530_Liver_ensg' #'lung' #'colon' #'breast' #'gse155468' #'GSE151530_Liver_ensg' # 'GSE131907_Lung_ensg'
#'HumanLiverCancerPatient2_filtered_ensg' #'HumanLungCancerPatient2_filtered_ensg' #'GSE97930_FrontalCortex_preprocessed' 
#'demo_train'(hPancread) #'c_data'(MS)

# 1. Load all 12 samples
data = ad.read_h5ad(f'./data/{DATA}.h5ad')


data.X = data.X.astype('float32')
# Unknown 처리
true_labels = data.obs['Type'].values
mask = true_labels != 'Unknown'

sc.pp.neighbors(data)
sc.tl.umap(data)
sc.tl.leiden(data, resolution=1.0, key_added='leiden_pca')

pca_leiden = data.obs['leiden_pca'].values.astype(str)
ari = adjusted_rand_score(true_labels[mask], pca_leiden[mask])
nmi = normalized_mutual_info_score(true_labels[mask], pca_leiden[mask])
print(ari, nmi)

sc.pl.umap(data, color=["leiden_pca"],
            size=20,
            show=False,      # 중요: 자동으로 그리지 않게 함
            title=f"{DATA}\nARI={ari:.2f}, NMI={nmi:.2f}"
    )

plt.tight_layout()
plt.savefig(f"./figure/{DATA}_embedding_pca.png", dpi=300)
plt.show()