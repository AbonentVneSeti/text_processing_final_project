import pandas as pd
from tqdm import tqdm
from .filters import (
    remove_duplicates,
    normalize_text,
    filter_case_and_yo,
    filter_trivial_pairs,
    filter_length_ratio,
    filter_by_length,
    filter_grammar_only_changes,
    filter_semantic_similarity,
)

STEP_MAP = {
    "normalize_text": normalize_text,
    "remove_duplicates": remove_duplicates,
    "filter_case_and_yo": filter_case_and_yo,
    "filter_trivial_pairs": filter_trivial_pairs,
    "filter_length_ratio": filter_length_ratio,
    "filter_by_length": filter_by_length,
    "filter_grammar_only_changes": filter_grammar_only_changes,
    "filter_semantic_similarity": filter_semantic_similarity,
}

def prepare_dataset(df: pd.DataFrame, preproc_config: dict) -> pd.DataFrame:
    steps = preproc_config.get("steps", [])
    with tqdm(total=len(steps), desc="Preprocessing", unit="step") as pbar:
        for step_name in steps:
            if step_name in STEP_MAP:
                func = STEP_MAP[step_name]
                params = preproc_config.get(step_name, {})
                pbar.set_description(f"Running {step_name} ({len(df)} pairs)")
                df = func(df, params)
                pbar.update(1)
            else:
                print(f"[Warning] Unknown step: '{step_name}', skipping.")
                pbar.update(1)
    return df