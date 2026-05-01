# dataset/create_dataset/tapaco.py
import pandas as pd
from datasets import load_dataset
from itertools import combinations

def load_or_create(config_section: dict) -> pd.DataFrame:
    """
    Загрузка русского подмножества TAPACO из Hugging Face.
    Ожидаемые поля конфига:
        - language: str (по умолчанию "ru")
        - pair_mode: str (по умолчанию "all_pairs")
            "all_pairs" – все возможные пары внутри paraphrase_set_id,
            "first_as_original" – первое предложение как оригинал, остальные как парафразы
    """
    language = config_section.get("language", "ru")
    pair_mode = config_section.get("pair_mode", "all_pairs")

    # Загружаем датасет (сплит 'train' содержит все данные)
    dataset = load_dataset("tapaco", language, split="train")

    # Преобразуем в pandas DataFrame, оставляем нужные колонки
    df_raw = dataset.to_pandas()[["paraphrase_set_id", "paraphrase"]]

    # Удаляем дубликаты предложений внутри одной группы
    df_unique = df_raw.drop_duplicates(subset=["paraphrase_set_id", "paraphrase"])

    # Группируем по идентификатору группы и формируем пары
    original_list = []
    paraphrase_list = []

    for _, group in df_unique.groupby("paraphrase_set_id"):
        sentences = group["paraphrase"].tolist()
        if len(sentences) < 2:
            continue  # одна строка не даёт пары

        if pair_mode == "first_as_original":
            original = sentences[0]
            for para in sentences[1:]:
                original_list.append(original)
                paraphrase_list.append(para)
        else:  # all_pairs
            for orig, para in combinations(sentences, 2):
                original_list.append(orig)
                paraphrase_list.append(para)
                # добавляем и обратную пару для симметричности (опционально)
                original_list.append(para)
                paraphrase_list.append(orig)

    result_df = pd.DataFrame({
        "original": original_list,
        "paraphrase": paraphrase_list
    })

    return result_df