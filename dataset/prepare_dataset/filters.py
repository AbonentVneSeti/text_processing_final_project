import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def remove_duplicates(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=['original', 'paraphrase']).reset_index(drop=True)

def filter_by_length(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.get("tokenizer", "cointegrated/rut5-base"))
    max_tokens = params.get("max_tokens", 128)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    mask = df['original'].apply(count_tokens) <= max_tokens
    mask &= df['paraphrase'].apply(count_tokens) <= max_tokens
    return df[mask].reset_index(drop=True)

def filter_edit_distance(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_ratio = params.get("min_edit_ratio", 0.1)
    max_ratio = params.get("max_edit_ratio", 0.9)

    def edit_ratio(row):
        dist = Levenshtein.distance(row['original'], row['paraphrase'])
        max_len = max(len(row['original']), len(row['paraphrase']))
        return dist / max_len if max_len > 0 else 0.0

    ratios = df.apply(edit_ratio, axis=1)
    mask = (ratios >= min_ratio) & (ratios <= max_ratio)
    return df[mask].reset_index(drop=True)

def filter_semantic_similarity(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    model_name = params.get("model", "sentence-transformers/LaBSE")
    threshold = params.get("threshold", 0.75)
    batch_size = params.get("batch_size", 32)
    model = SentenceTransformer(model_name)

    def compute_sim_batched(texts1, texts2):
        emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=False)
        emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=False)

        sim = np.sum(emb1 * emb2, axis=1) / (np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1))
        return sim

    sims = compute_sim_batched(df['original'].tolist(), df['paraphrase'].tolist())
    mask = sims >= threshold
    return df[mask].reset_index(drop=True)