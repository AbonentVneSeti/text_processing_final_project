import pandas as pd
from deep_translator import GoogleTranslator  # pip install deep-translator
from tqdm import tqdm

def load_or_create(config_section: dict) -> pd.DataFrame:
    input_file = config_section.get("input_file", "data/sentences.txt")
    from_lang = config_section.get("from_lang", "ru")
    to_lang = config_section.get("to_lang", "en")
    api = config_section.get("api", "googletrans")

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    originals = []
    paraphrases = []
    translator_to = GoogleTranslator(source=from_lang, target=to_lang)
    translator_back = GoogleTranslator(source=to_lang, target=from_lang)

    for sent in tqdm(sentences, desc="Back-translating"):
        try:
            trans_en = translator_to.translate(sent)
            trans_ru = translator_back.translate(trans_en)
            originals.append(sent)
            paraphrases.append(trans_ru)
        except Exception as e:
            print(f"Error on '{sent}': {e}")
            continue

    df = pd.DataFrame({'original': originals, 'paraphrase': paraphrases})
    return df