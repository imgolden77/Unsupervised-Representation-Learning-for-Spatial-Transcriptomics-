# 🚀 Running Experiments

## Dataset Preparation

All datasets used in our tutorials are collected from previous publications and we provide the references in [`../data/README.md`](https://github.com/imgolden77/Unsupervised-Representation-Learning-for-Spatial-Transcriptomics-/blob/main/data/README.md).

Before running the tutorial, please download datasets from our [dropbox](https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=0) and placed h5ad datasets in the `../data` folder.

## Customized Dataset

The customized dataset can now be easily processed with `CellPLM.pipeline` module. Please refer to the tutorial of each downstream task.

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