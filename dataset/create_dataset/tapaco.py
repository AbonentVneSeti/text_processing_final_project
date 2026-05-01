import pandas as pd
from datasets import load_dataset
from itertools import combinations

def load_or_create(config_section: dict) -> pd.DataFrame:
    language = config_section.get("language", "ru")
    pair_mode = config_section.get("pair_mode", "all_pairs")

    dataset = load_dataset("tapaco", language, split="train")

    df_raw = dataset.to_pandas()[["paraphrase_set_id", "paraphrase"]]

    df_unique = df_raw.drop_duplicates(subset=["paraphrase_set_id", "paraphrase"])

    original_list = []
    paraphrase_list = []

    for _, group in df_unique.groupby("paraphrase_set_id"):
        sentences = group["paraphrase"].tolist()
        if len(sentences) < 2:
            continue

        if pair_mode == "first_as_original":
            original = sentences[0]
            for para in sentences[1:]:
                original_list.append(original)
                paraphrase_list.append(para)
        else:
            for orig, para in combinations(sentences, 2):
                original_list.append(orig)
                paraphrase_list.append(para)

                original_list.append(para)
                paraphrase_list.append(orig)

    result_df = pd.DataFrame({
        "original": original_list,
        "paraphrase": paraphrase_list
    })

    return result_df