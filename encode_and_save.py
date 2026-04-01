import scanpy as sc
import anndata as ad
import numpy as np
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

def load_cohort_data(disease: str, n_top_genes: int):
    """
    Parameters
    ----------
    disease : str
        Disease abbreviation (e.g. "UC", "CD", "AD", "T1D", "PSO").

    Returns
    -------
    ad.AnnData
        AnnData with raw FPKM in ``.X``.  ``.obs["disease"]`` contains the
        condition label; healthy controls are marked ``"Control"``.
    """

    path = hf_hub_download(
        repo_id="ScientaLab/bulk-rna-immuno-inflammation-cohorts",
        filename=f"{disease}/dataset.h5ad",   # filename may vary
        repo_type="dataset"
    )

    adata = ad.read_h5ad(path)

    # Subset to 2,000 highly variable genes for efficiency
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable].copy()

    return adata

def encode_and_save_data(model, tokenizer, adata, out_fn):
    # Encode (gene symbols auto-converted, preprocessing applied, GPU used if available)
    embeddings = model.encode_anndata(tokenizer, adata)
    
    embeddings = embeddings.cpu().numpy()
    np.save(out_fn, embeddings)


if __name__ == "__main__":
    # Load model
    model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)

    disease_list = ['AD', 'CD', 'PSO', 'T1D', 'UC']

    for disease in disease_list:
        output_fn = "data/emb/{disease}_eva_rna.npy"

        adata = load_cohort_data(disease)
        encode_and_save_data(model, tokenizer, adata, output_fn)

