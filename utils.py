import yaml
import pandas as pd
from torch.utils.data import DataLoader, Dataset as TorchDataset

class ParaphraseDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return {
            "original": self.df.loc[idx, "original"],
            "paraphrase": self.df.loc[idx, "paraphrase"]
        }

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_dataloaders(df: pd.DataFrame, model_config: dict, split_ratios=(0.8, 0.1, 0.1), seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    batch_size = model_config.get("batch_size", 8)

    class Wrapper:
        def __init__(self, dataframe, batch_size):
            self.dataset = dataframe
            self.batch_size = batch_size
        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset.iloc[i:i+self.batch_size]
                yield batch

    train_loader = Wrapper(train_df, batch_size)
    val_loader = Wrapper(val_df, batch_size)
    test_loader = Wrapper(test_df, batch_size)
    return train_loader, val_loader, test_loader