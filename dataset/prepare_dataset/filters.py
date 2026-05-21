import re
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch


def remove_duplicates(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['original', 'paraphrase'])
    mask = df['original'].str.strip() != df['paraphrase'].str.strip()
    return df[mask].reset_index(drop=True)


def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from .cleaners import normalize_text as _normalize
    return _normalize(df, params)


def filter_case_and_yo(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    def normalize(s):
        return s.strip().lower().replace('ё', 'е')
    mask = df['original'].apply(normalize) != df['paraphrase'].apply(normalize)
    return df[mask].reset_index(drop=True)


def filter_trivial_pairs(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_edit_ratio = params.get("min_edit_ratio", 0.15)
    max_edit_ratio = params.get("max_edit_ratio", 0.85)
    min_len = params.get("min_len", 5)
    max_len = params.get("max_len", 128)

    def edit_ratio(row):
        dist = Levenshtein.distance(row['original'], row['paraphrase'])
        max_len_chars = max(len(row['original']), len(row['paraphrase']))
        return dist / max_len_chars if max_len_chars > 0 else 0.0

    ratios = df.apply(edit_ratio, axis=1)
    mask = (ratios >= min_edit_ratio) & (ratios <= max_edit_ratio)
    df = df[mask].copy()

    len_orig = df['original'].str.split().str.len()
    len_para = df['paraphrase'].str.split().str.len()
    mask = (
        (len_orig >= min_len) & (len_orig <= max_len) &
        (len_para >= min_len) & (len_para <= max_len)
    )
    return df[mask].reset_index(drop=True)


def filter_length_ratio(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_ratio = params.get("min_ratio", 0.6)
    max_ratio = params.get("max_ratio", 1.7)

    len_orig = df['original'].str.split().str.len().replace(0, np.nan)
    len_para = df['paraphrase'].str.split().str.len()
    ratio = len_para / len_orig
    mask = (ratio >= min_ratio) & (ratio <= max_ratio)
    return df[mask].reset_index(drop=True)


def filter_by_length(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    tokenizer_name = params.get("tokenizer", "cointegrated/rut5-base")
    max_tokens = params.get("max_tokens", 128)
    min_tokens = params.get("min_tokens", 5)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    orig_lens = df['original'].apply(count_tokens)
    para_lens = df['paraphrase'].apply(count_tokens)
    mask = (
        orig_lens.between(min_tokens, max_tokens) &
        para_lens.between(min_tokens, max_tokens)
    )
    return df[mask].reset_index(drop=True)


def filter_grammar_only_changes(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    import pymorphy3
    from collections import Counter

    morph = pymorphy3.MorphAnalyzer()
    min_gram_ratio = params.get("min_gram_ratio", 0.8)

    def get_lemma_and_grammar(text):
        tokens = re.findall(r'\w+', text.lower())
        result = []
        for token in tokens:
            parsed = morph.parse(token)[0]
            lemma = parsed.normal_form
            grammar = str(parsed.tag.POS)
            result.append((lemma, grammar))
        return result

    def is_grammar_only(row):
        orig_items = get_lemma_and_grammar(row['original'])
        para_items = get_lemma_and_grammar(row['paraphrase'])

        orig_lemmas = Counter(l for l, _ in orig_items)
        para_lemmas = Counter(l for l, _ in para_items)
        if orig_lemmas != para_lemmas:
            return False

        orig_sorted = sorted(orig_items, key=lambda x: x[0])
        para_sorted = sorted(para_items, key=lambda x: x[0])
        changes = sum(1 for (_, g1), (_, g2) in zip(orig_sorted, para_sorted) if g1 != g2)
        total = len(orig_sorted)
        return total > 0 and (changes / total) >= min_gram_ratio

    mask = df.apply(is_grammar_only, axis=1)
    return df[~mask].reset_index(drop=True)


def filter_semantic_similarity(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    model_name = params.get("model", "sentence-transformers/LaBSE")
    min_threshold = params.get("min_threshold", 0.85)
    max_threshold = params.get("max_threshold", 0.97)
    batch_size = params.get("batch_size", 64)

    model = SentenceTransformer(model_name)

    texts1 = df['original'].tolist()
    texts2 = df['paraphrase'].tolist()

    emb1 = model.encode(texts1, batch_size=batch_size, show_progress_bar=True)
    emb2 = model.encode(texts2, batch_size=batch_size, show_progress_bar=True)

    norms1 = np.linalg.norm(emb1, axis=1)
    norms2 = np.linalg.norm(emb2, axis=1)
    sims = np.sum(emb1 * emb2, axis=1) / (norms1 * norms2)

    mask = (sims >= min_threshold) & (sims <= max_threshold)
    return df[mask].reset_index(drop=True)