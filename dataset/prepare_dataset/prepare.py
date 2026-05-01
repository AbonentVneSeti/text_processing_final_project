import pandas as pd
from tqdm import tqdm
from .cleaners import normalize_text
from .filters import (
    remove_duplicates,
    filter_by_length,
    filter_edit_distance,
    filter_semantic_similarity
)

STEP_MAP = {
    "remove_duplicates": remove_duplicates,
    "normalize_text": normalize_text,
    "filter_by_length": filter_by_length,
    "filter_edit_distance": filter_edit_distance,
    "filter_semantic_similarity": filter_semantic_similarity,
}

def prepare_dataset(df: pd.DataFrame, preproc_config: dict) -> pd.DataFrame:
    steps = preproc_config.get("steps", [])
    # Общий прогресс-бар по шагам
    with tqdm(total=len(steps), desc="Preprocessing", unit="step") as pbar:
        for step_name in steps:
            if step_name in STEP_MAP:
                func = STEP_MAP[step_name]
                params = preproc_config.get(step_name, {})
                pbar.set_description(f"Running {step_name}")
                df = func(df, params)
            pbar.update(1)
    return df