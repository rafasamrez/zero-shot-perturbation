# Loading data

# Loading dataset from huggingface (bulk RNA immuno-inflammation cohorts)
from datasets import load_dataset
ds = load_dataset("ScientaLab/bulk-rna-immuno-inflammation-cohorts")

# Load RNA-seq cohort
from huggingface_hub import hf_hub_download
disease = "UC" # Change abbreviation according to the disease

path = hf_hub_download(
    repo_id="ScientaLab/bulk-rna-immuno-inflammation-cohorts",
    filename=f"{disease}/dataset.h5ad",   # filename may vary
    repo_type="dataset"
)

import anndata as ad
adata = ad.read_h5ad(path)

## Find indexes for disease and control
import numpy as np
control_idx = np.where(adata.obs.values[:,1] == 'Control')[0]
disease_idx = np.where(adata.obs.values[:,1] != 'Control')[0]

## Raw FPKM 
control_seq = adata.X[control_idx]
disease_seq = adata.X[disease_idx]

# Loading drug-disease efficacy matrix benchmark
import pandas as pd
data = pd.read_csv('data/benchmark_drug_target_disease_matrix.csv')