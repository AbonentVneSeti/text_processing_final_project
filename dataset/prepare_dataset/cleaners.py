import re
import pandas as pd


def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    trim = params.get("trim", True)
    remove_brackets = params.get("remove_brackets", True)

    def clean(text):
        if trim:
            text = text.strip()
        if remove_brackets:
            text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    df = df.copy()
    df['original'] = df['original'].apply(clean)
    df['paraphrase'] = df['paraphrase'].apply(clean)
    return df