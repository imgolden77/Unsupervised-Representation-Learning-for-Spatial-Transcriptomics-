# Unsupervised Representation learning for Spatial transcriptomics : Extensions and Evaluation of CellPLM

A Master of Science Dissertation submitted to The University of Manchester.

## 🌟 Overview
<p align="center">
  <img src="image/CellPLM Architecture.png" alt="CellPLM banner" width="600"/>
</p>

This repository contains the code and resources related to the Master's dissertation, **"Unsupervised representation learning for spatial transcriptomics."** The work focuses on extensively **revisiting, extending, and evaluating** the **CellPLM** (Cell Pre-trained Language Model) foundation model, a Transformer-based architecture designed for single-cell and spatial transcriptomics (ST) data analysis.

The core objective was to deepen the theoretical understanding of CellPLM's components (Cell Language Model, Flowformer, Gaussian Mixture Latent Space) and expand its practical application and scalability for downstream tasks like **cell clustering**, **cell type annotation**, and **ST imputation**.

## ✨ Key Project Achievements

The project makes several crucial contributions, particularly in the area of fine-tuning CellPLM for optimal performance and efficiency.

  * **Novel Clustering Fine-tuning Pipeline:** We designed and implemented a systematic fine-tuning pipeline for cell embedding clustering, comparing Self-supervised, Supervised Contrastive, and Fully Supervised learning strategies.
  * **Scalable Supervised Contrastive Learning (SupConLoss):** Introduced and validated a supervised contrastive (SupConLoss) head for fine-tuning. This method achieves clustering performance **competitive with Cross-Entropy Loss** while reducing training time by **$30\times$ to $300\times$**. This establishes a highly scalable alternative for large-scale or weakly labeled single-cell datasets.
  * **Extended Empirical Evaluation:** Conducted extensive experiments on a wide array of scRNA-seq and ST datasets (e.g., Breast Cancer, Aorta, DLPFC Visium, MERFISH mouse brain2) that were not fully covered in the original CellPLM work.
  * **Codebase Extension:** Publicly-available implementation was extended to support crucial missing functionalities, including:
    1.  Fine-tuned cell embedding clustering
    2.  Zero-shot inference for cell type annotation

## 🛠️ Model Architecture Highlights

The core architecture is based on the pre-trained CellPLM model, utilizing the following components:

  * **Cell Language Model:** Extends the gene-centric view by explicitly modeling **cell-to-cell dependencies**, treating cells as tokens to capture biologically crucial intercellular relationships.
  * **Flowformer Encoder:** Employs **Flowformer**, a variant of the Transformer with $O(nd^2)$ complexity, making it more efficient than standard $O(n^2d)$ attention for the long sequence lengths characteristic of single-cell and ST datasets.
  * **Gaussian Mixture Latent Space (GMVAE):** Uses a GMVAE prior to better capture the **heterogeneous cell groups** and introduce an inductive bias, generating smoother and more informative cell representations.
  * **Batch-Aware Decoder:** Incorporates batch-specific embeddings to absorb technical variation, ensuring the latent space remains **biologically meaningful and batch-invariant**.

## 📊 Evaluation Results Summary

| Task | Datasets | Key Finding |
| :--- | :--- | :--- |
| **Cell Embedding Clustering** | DLPFC, Mouse Brain2, Breast, Aorta | **CellPLM (Zero-shot)** consistently **outperforms PCA** on all datasets. **SupConLoss** fine-tuning provides a **superior speed-quality trade-off** compared to Cross-Entropy Loss (e.g., $30\times$ to $300\times$ faster training). |
| **Cell Type Annotation** | DLPFC, Mouse Brain2, Liver/Lung Cancer, Aorta | **Fine-tuning is essential**; zero-shot accuracy is near random. CellPLM demonstrates **strong generalization** on unseen scRNA-seq data (Aorta, Lung, Colorectal cancer) with F1-scores above 0.95. |
| **ST Imputation** | DLPFC Visium, MERFISH Mouse Brain2 | The utility of scRNA-seq reference data is **context-dependent**. It is beneficial for extremely sparse datasets like MERFISH (155 genes) but provides limited advantage for richer datasets like DLPFC Visium (33,538 genes). |

See [thesis](https://github.com/imgolden77/Unsupervised-Representation-Learning-for-Spatial-Transcriptomics-/blob/main/juhaim_thesis_end.pdf) and the `image/` directory for full experimental results and illustrations.

<p align="center">
  <img src="image/dlpfc_img.jpeg" alt="..." width="800"/>
</p>

<p align="center">
  <img src="image/umap_breast_aorta.jpg" alt="..." width="800"/>
</p>

## Quick start

**Install dependencies**

```bash
# Recommended: Python 3.9, CUDA >= 11.7
pip install -r requirements.txt
```

**Or using conda (recommended for reproducibility):**

```bash
conda create -n cellplm python=3.9 -y
conda activate cellplm
conda install cudatoolkit=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data Preparation

The datasets used in this work are publicly available and listed in the thesis:

  * **scRNA-seq:** Breast Cancer, Colorectal Cancer, Lung, Liver Cancer, Lung Cancer, Aorta, Frontal Cortex, Mouse Brain.
  * **ST:** DLPFC Visium (12 samples), MERFISH Mouse Brain2 (5 samples).

The preprocessing functions from the CellPLM framework (`common_preprocess` and `transcriptomics_dataset`) were used to standardize `AnnData` objects and filter the gene list against the pre-trained set.

## Pretrained CellPLM Model Checkpoints
The checkpoint can be acquired from [dropbox](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0). 
[10/10/2023] The latest version is `20230926_85M`.

## 🚀 Running Experiments

Detailed scripts and configuration files for reproducing the main fine-tuning results (SupConLoss vs. Cross-Entropy) are located in the `tutorials/` directory.

### 1\. Fine-tuned Cell Embedding Clustering (SupConLoss vs. Cross-Entropy)

To reproduce the key comparison of fine-tuning efficiency and performance on a dataset like Aorta:

  * **Configuration:** Adjust hyperparameters in the relevant pipeline file, `CellPLM/CellPLM/pipeline/cell_embedding_fit.py` for SupConLoss, `CellPLM/CellPLM/pipeline/cell_type_annotation.py` for Cross-Entropy.
  * **Execution:**
    ```bash
    # Run with Supervised Contrastive Loss
    python embedding_fit.py

    # Run with Cross-Entropy Loss
    python annotation_fit_emb.py 
    ```
  * **Metrics:** Clustering metrics (ARI, NMI) and training time will be reported as shown in Table 4 and Table 5 of the thesis.

### 2\. Cell Type Annotation

  * **Configuration:** The best performing setting used an **Autoencoder** as the latent model, with 3,000 highly variable genes (HVGs) and no positional encoding (PE). Adjust hyper parameters in `CellPLM/CellPLM/pipeline/cell_type_annotation.py`
  * **Execution (Example: DLPFC Layer Segmentation):**
    ```bash
    python annotation_fit.py 
    ```
  * **Metrics:** Accuracy and Macro $F_{1}$ scores will be reported.

### 3\. Spatial Transcriptomics Imputation

  * **Configuration:** Adjust hyper parameters in `CellPLM/CellPLM/pipeline/imputation.py`
  * **Execution (Example: DLPFC):**
    ```bash
    # Fine-tuning with scRNA-seq reference data
    python imputation_fit.py 

    # Zero-shot inference (Pre-trained parameters only)
    python imputation_zeroshot.py 
    ```
  * **Metrics:** Results include MSE, RMSE, MAE, Pearson's Correlation Coefficient (PCC), and Cosine similarity.

## Repository layout (important files)

- `CellPLM/` — core Python package (models, layers, utils)
- `ckpt/` — model checkpoints (`.ckpt`) and corresponding `.config.json` files
- `data/` — datasets (raw / preprocessed samples)
- `image/` — figures used in the paper and experiment visualizations
- `tutorials/` — example notebooks and finetuning
- `requirements.txt` — full dependency list
- `juhaim_thesis_end.pdf` — thesis PDF included in repository

## 🤝 Acknowledgements

This Master project is based on [CellPLM: Pre-training of Cell Language Model Beyond Single Cells](https://openreview.net/forum?id=BKXvPDekud).

![Paper](https://img.shields.io/badge/Paper-ICLR24-brightgreen?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DBKXvPDekud)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

Cite the paper as:

```
@article{wen2023cellplm,
  title={CellPLM: Pre-training of Cell Language Model Beyond Single Cells},
  author={Wen, Hongzhi and Tang, Wenzhuo and Dai, Xinnan and Ding, Jiayuan and Jin, Wei and Xie, Yuying and Tang, Jiliang},
  journal={bioRxiv},
  pages={2023--10},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```


