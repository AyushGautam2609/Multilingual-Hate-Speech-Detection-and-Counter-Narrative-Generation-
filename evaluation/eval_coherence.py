import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

def evaluate_coherence(df, hs_col, cn_cols, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    """
    Computes dense semantic cosine similarity between hate speech prompts and counter-narratives.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device)
    
    hs_embeddings = embedder.encode(
        df[hs_col].fillna("").astype(str).tolist(), 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    results = {}
    for col in cn_cols:
        cn_embeddings = embedder.encode(
            df[col].fillna("").astype(str).tolist(), 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        cos_sims = (hs_embeddings * cn_embeddings).sum(dim=1).cpu().numpy()
        results[f"{col}_coherence"] = [round(float(s), 4) for s in cos_sims]
        
    return pd.DataFrame(results)
