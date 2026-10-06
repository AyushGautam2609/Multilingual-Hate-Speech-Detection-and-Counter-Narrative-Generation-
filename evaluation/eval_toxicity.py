import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_muril_toxicity(df, text_cols, model_name="google/muril-base-cased"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    results = {}
    for col in text_cols:
        scores = []
        for text in df[col].fillna("").astype(str):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                prob = torch.softmax(logits, dim=-1)[0][-1].item()
            scores.append(round(prob, 4))
        results[f"{col}_toxicity"] = scores
    return pd.DataFrame(results)
