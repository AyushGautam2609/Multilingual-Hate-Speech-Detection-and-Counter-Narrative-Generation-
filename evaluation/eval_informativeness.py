import numpy as np
import pandas as pd

def calculate_ridf(doc_term_matrix, term_doc_freq, total_docs):
    cf = doc_term_matrix.sum(axis=0)
    expected_df = total_docs * (1.0 - np.exp(-cf / total_docs))
    actual_df = np.maximum(term_doc_freq, 1e-9)
    idf = -np.log(actual_df / total_docs)
    predicted_idf = -np.log(np.maximum(expected_df, 1e-9) / total_docs)
    return idf - predicted_idf
