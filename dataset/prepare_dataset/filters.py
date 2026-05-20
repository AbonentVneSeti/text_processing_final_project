import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import re

def remove_duplicates(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['original', 'paraphrase'])
    mask = df['original'].str.strip() != df['paraphrase'].str.strip()
    return df[mask].reset_index(drop=True)

def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    trim = params.get("trim", True)
    remove_brackets = params.get("remove_brackets", True)

    def clean(text):
        if trim:
            text = text.strip()
        if remove_brackets:
            text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', text)
        return text

    df['original'] = df['original'].apply(clean)
    df['paraphrase'] = df['paraphrase'].apply(clean)
    return df

def filter_trivial_pairs(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_edit_ratio = params.get("min_edit_ratio", 0.05)
    max_edit_ratio = params.get("max_edit_ratio", 0.9)
    min_len = params.get("min_len", 3)
    max_len = params.get("max_len", 128)

    def edit_ratio(row):
        dist = Levenshtein.distance(row['original'], row['paraphrase'])
        max_len_chars = max(len(row['original']), len(row['paraphrase']))
        return dist / max_len_chars if max_len_chars > 0 else 0.0

    ratios = df.apply(edit_ratio, axis=1)
    mask = (ratios >= min_edit_ratio) & (ratios <= max_edit_ratio)

    len_orig = df['original'].str.split().str.len()
    len_para = df['paraphrase'].str.split().str.len()
    mask &= (len_orig >= min_len) & (len_orig <= max_len)
    mask &= (len_para >= min_len) & (len_para <= max_len)

    return df[mask].reset_index(drop=True)

def filter_by_length(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    tokenizer_name = params.get("tokenizer", "cointegrated/rut5-base")
    max_tokens = params.get("max_tokens", 128)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    mask = df['original'].apply(count_tokens) <= max_tokens
    mask &= df['paraphrase'].apply(count_tokens) <= max_tokens
    return df[mask].reset_index(drop=True)

def filter_length_ratio(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_ratio = params.get("min_ratio", 0.5)
    max_ratio = params.get("max_ratio", 2.0)

    orig_len = df['original'].str.len()
    para_len = df['paraphrase'].str.len()
    ratio = para_len / orig_len
    mask = (ratio >= min_ratio) & (ratio <= max_ratio)
    return df[mask].reset_index(drop=True)

def filter_semantic_similarity(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    model_name = params.get("model", "sentence-transformers/LaBSE")
    min_threshold = params.get("min_threshold", 0.7)
    max_threshold = params.get("max_threshold", 0.98)
    batch_size = params.get("batch_size", 32)

    model = SentenceTransformer(model_name)

    def compute_sim_batched(texts1, texts2):
        emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=False)
        emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=False)
        sim = np.sum(emb1 * emb2, axis=1) / (
            np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
        )
        return sim

    sims = compute_sim_batched(df['original'].tolist(), df['paraphrase'].tolist())
    mask = (sims >= min_threshold) & (sims <= max_threshold)
    return df[mask].reset_index(drop=True)

def filter_case_and_yo(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    def normalize(s):
        return s.strip().lower().replace('ё', 'е')
    mask = df['original'].apply(normalize) != df['paraphrase'].apply(normalize)
    return df[mask].reset_index(drop=True)