import pandas as pd

def load_or_create(config_section: dict) -> pd.DataFrame:
    files = config_section.get("files", [])
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[['original', 'paraphrase']].dropna()
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    return merged