import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("data/paraphrases_clean.parquet")

df['orig_chars'] = df['original'].str.len()
df['para_chars'] = df['paraphrase'].str.len()

df['orig_words'] = df['original'].str.split().str.len()
df['para_words'] = df['paraphrase'].str.split().str.len()

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rut5-base")
df['orig_tokens'] = df['original'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
df['para_tokens'] = df['paraphrase'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))

def compute_edit(row):
    dist = Levenshtein.distance(row['original'], row['paraphrase'])
    max_len = max(len(row['original']), len(row['paraphrase']))
    return dist, dist / max_len if max_len > 0 else 0.0

edit_data = df.apply(compute_edit, axis=1, result_type='expand')
df['edit_dist'] = edit_data[0]
df['edit_ratio'] = edit_data[1]

df['len_ratio'] = df['para_words'] / df['orig_words'].replace(0, np.nan)


stats = {}
for col in ['orig_chars', 'para_chars', 'orig_words', 'para_words', 'orig_tokens', 'para_tokens',
            'edit_dist', 'edit_ratio', 'len_ratio']:
    stats[col] = {
        'mean': df[col].mean(),
        'median': df[col].median(),
        'min': df[col].min(),
        'max': df[col].max(),
        'std': df[col].std()
    }

stats_df = pd.DataFrame(stats).T
print(stats_df)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
sns.histplot(df['orig_words'], bins=50, ax=axes[0,0], discrete = True)
axes[0,0].set_title('Длина оригинала (слова)')
sns.histplot(df['para_words'], bins=50, ax=axes[0,1], discrete = True)
axes[0,1].set_title('Длина парафраза (слова)')
sns.histplot(df['orig_tokens'], bins=50, ax=axes[0,2], discrete = True)
axes[0,2].set_title('Длина оригинала (токены)')
sns.histplot(df['edit_ratio'], bins=50, ax=axes[1,0])
axes[1,0].set_title('Edit ratio')
sns.histplot(df['len_ratio'], bins=50, ax=axes[1,1])
axes[1,1].set_title('Соотношение длин (парафраз/оригинал)')
plt.tight_layout()
plt.savefig('dataset_distributions.png', dpi=150)
plt.show()


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')

sample = df.sample(n=10000, random_state=42)
orig_emb = model.encode(sample['original'].tolist(), show_progress_bar=True)
para_emb = model.encode(sample['paraphrase'].tolist(), show_progress_bar=True)

sims = np.sum(orig_emb * para_emb, axis=1) / (np.linalg.norm(orig_emb, axis=1) * np.linalg.norm(para_emb, axis=1))
print(f"Средняя семантическая близость: {sims.mean():.4f} (±{sims.std():.4f})")

plt.figure(figsize=(6,4))
sns.histplot(sims, bins=50)
plt.title('Распределение косинусного сходства пар (LaBSE)')
plt.xlabel('Cosine similarity')
plt.savefig('semantic_similarity_dist.png', dpi=150)
plt.show()