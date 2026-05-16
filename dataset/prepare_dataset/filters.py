import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re


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


def filter_trivial_pairs(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_edit_ratio = params.get("min_edit_ratio", 0.2)
    min_len = params.get("min_len", 6)
    max_len = params.get("max_len", 50)

    mask = df.apply(lambda row: (
        Levenshtein.distance(row['original'], row['paraphrase']) / max(len(row['original']), len(row['paraphrase']), 1)
    ) >= min_edit_ratio, axis=1)
    df = df[mask]

    df = df[df['original'].str.split().str.len().between(min_len, max_len)]
    df = df[df['paraphrase'].str.split().str.len().between(min_len, max_len)]
    return df.reset_index(drop=True)


def filter_near_duplicates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    max_jaccard = params.get("max_jaccard", 0.9)

    def tokenize(text):
        return set(re.findall(r'\w+', text.lower()))

    def jaccard_similarity(orig, para):
        tok_orig = tokenize(orig)
        tok_para = tokenize(para)
        if not tok_orig or not tok_para:
            return 0.0
        return len(tok_orig & tok_para) / len(tok_orig | tok_para)

    mask = df.apply(lambda row: jaccard_similarity(row['original'], row['paraphrase']) < max_jaccard, axis=1)
    return df[mask].reset_index(drop=True)



# GENDER_NEUTRAL = {"ещё", "еще"}

# def get_gender_markers(text):
#     words = re.findall(r'[а-яё]+', text.lower())
#     markers = []
#     for w in words:
#         if w in GENDER_NEUTRAL:
#             continue
#         if w.endswith(('ая', 'яя')):
#             markers.append((w[:-2], 'fem'))
#         elif w.endswith(('ый', 'ий', 'ой')):
#             stem = re.sub(r'(ый|ий|ой)$', '', w)
#             markers.append((stem, 'masc'))
#         elif w.endswith('ла') and len(w) > 3:
#             markers.append((w[:-2], 'fem'))
#         elif w.endswith('л') and not w.endswith(('ал', 'ел', 'ил', 'ол', 'ул', 'ыл')): 
#             pass
#     return markers

# def filter_gender_swaps(df: pd.DataFrame, params: dict) -> pd.DataFrame:
#     max_ratio = params.get("max_gender_diff_ratio", 0.0)

#     def gender_swapped(row):
#         markers_orig = get_gender_markers(row['original'])
#         markers_para = get_gender_markers(row['paraphrase'])

#         orig_dict = {stem: gen for stem, gen in markers_orig}
#         para_dict = {stem: gen for stem, gen in markers_para}

#         common = set(orig_dict.keys()) & set(para_dict.keys())
#         if not common:
#             return False

#         changed = sum(1 for stem in common if orig_dict[stem] != para_dict[stem])
#         total_words = min(len(markers_orig), len(markers_para))
#         if total_words == 0:
#             return False
#         return (changed / total_words) > max_ratio

#     mask = df.apply(gender_swapped, axis=1)
#     return df[~mask].reset_index(drop=True)



def get_lemma_and_grammar(text, morph):
    words = re.findall(r'[а-яё]+', text.lower())
    result = []
    for w in words:
        p = morph.parse(w)[0]
        gram = {g for g in p.tag.grammemes if g in {'masc','femn','neut','plur','sing',
                                                    '1per','2per','3per'}}
        result.append((p.normal_form, frozenset(gram)))
    return result

def filter_grammar_only_changes(df: pd.DataFrame, params: dict) -> pd.DataFrame:

    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()

    max_ratio = params.get("max_ratio", 0.0)

    from collections import Counter

    def is_grammar_only(row):
        orig_items = get_lemma_and_grammar(row['original'], morph)
        para_items = get_lemma_and_grammar(row['paraphrase'], morph)

        orig_lemmas = Counter(l for l, _ in orig_items)
        para_lemmas = Counter(l for l, _ in para_items)
        if orig_lemmas != para_lemmas:
            return False

        orig_sorted = sorted(orig_items, key=lambda x: x[0])
        para_sorted = sorted(para_items, key=lambda x: x[0])
        changes = sum(1 for (_, g1), (_, g2) in zip(orig_sorted, para_sorted) if g1 != g2)
        total = len(orig_sorted)
        return total > 0 and (changes / total) > max_ratio

    mask = df.apply(is_grammar_only, axis=1)
    return df[~mask].reset_index(drop=True)