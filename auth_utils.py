import numpy as np

def get_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    a = emb1.astype(np.float64)
    b = emb2.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)