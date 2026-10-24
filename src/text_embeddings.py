"""Sentence embeddings of article text, used as an item-tower input (helps cold-start items)."""

import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, ~23M params: fast on CPU


def article_texts(articles: pd.DataFrame) -> list[str]:
    """prod_name plus the free-text description; ~0.4% of articles have no description."""
    name = articles["prod_name"].fillna("").astype(str)
    desc = articles["detail_desc"].fillna("").astype(str)
    return (name + ". " + desc).str.strip(". ").tolist()


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_articles(articles: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=pick_device())
    emb = model.encode(
        article_texts(articles), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return emb.astype(np.float16)  # halves the file; precision is irrelevant for an input feature
