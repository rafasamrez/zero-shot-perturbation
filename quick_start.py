import scanpy as sc
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)

# Load example dataset (2,700 PBMCs, raw counts)
adata = sc.datasets.pbmc3k()

# Subset to 2,000 highly variable genes for efficiency
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
adata = adata[:, adata.var.highly_variable].copy()

# Encode (gene symbols auto-converted, preprocessing applied, GPU used if available)
embeddings = model.encode_anndata(tokenizer, adata)
adata.obsm["X_eva"] = embeddings

