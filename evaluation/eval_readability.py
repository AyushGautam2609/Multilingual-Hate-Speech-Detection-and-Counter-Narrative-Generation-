import re

def compute_sinha_readability(text):
    words = [w for w in re.split(r'\s+', text.strip()) if w]
    if not words:
        return 0.0, 0.0
    num_words = len(words)
    num_chars = sum(len(w) for w in words)
    L = num_chars / num_words
    
    rh1 = 0.4 * L + 0.05 * num_words
    rh2 = 0.55 * L + 0.10 * num_words
    return round(rh1, 4), round(rh2, 4)
