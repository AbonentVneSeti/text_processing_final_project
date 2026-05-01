import re
import pandas as pd

def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    lower = params.get("lower", True)
    trim = params.get("trim", True)
    remove_brackets = params.get("remove_brackets", True)

    def clean(text):
        if lower:
            text = text.lower()
        if trim:
            text = text.strip()
        if remove_brackets:
            text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', text)
        return text

    df['original'] = df['original'].apply(clean)
    df['paraphrase'] = df['paraphrase'].apply(clean)
    return df