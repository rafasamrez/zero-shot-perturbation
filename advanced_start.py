import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# Gene IDs must be NCBI GeneIDs as strings
gene_ids = ["7157", "675", "672"]  # TP53, BRCA2, BRCA1
expression_values = [5.5, 3.2, 4.1]  # log1p-normalized

inputs = tokenizer(gene_ids, expression_values, padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)

sample_embedding = outputs.cls_embedding     # (1, 256)
gene_embeddings = outputs.gene_embeddings   # (1, 3, 256)

# Batch Processing
batch_gene_ids = [
    ["7157", "675", "672"],
    ["7157", "1956", "5290"],
]
batch_expression = [
    [5.5, 3.2, 4.1],
    [2.1, 6.3, 1.8],
]

inputs = tokenizer(batch_gene_ids, batch_expression, padding=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)
sample_embeddings = outputs.cls_embedding  # (2, 256)

# Expression Decoder 
''' EVA-RNA includes a pre-trained deterministic expression decoder that maps gene embeddings back to predicted expression values.'''
with torch.inference_mode():
    # Encode
    output = model.encode(**inputs)
    # output.cls_embedding   — sample-level embedding (batch, hidden_size)
    # output.gene_embeddings — per-gene embeddings (batch, n_genes, hidden_size)

    # Decode expression values
    predicted_expression = model.decode(output.gene_embeddings)
    # predicted_expression — (batch, n_genes)

# GPU and Precision
'''EVA-RNA automatically applies mixed precision for optimal performance:

    Ampere+ GPUs (A100, H100, RTX 30/40 series): bfloat16
    Older CUDA GPUs (V100, RTX 20 series): float16
    CPU: full precision (float32)

No manual torch.autocast() is needed.
'''

# Disabling Automatic Mixed Precision
''' For advanced use cases requiring manual precision control, pass autocast=False. This only takes effect when flash attention is not active (i.e., on older GPUs or when flash attention is not installed):
'''
model = model.to("cuda").eval()

with torch.inference_mode():
    # Disable automatic mixed precision (ignored when flash attention is active)
    outputs = model(**inputs, autocast=False)

    # Or via sample_embedding
    embedding = model.sample_embedding(
        gene_ids=gene_ids,
        expression_values=values,
        autocast=False,
    )

