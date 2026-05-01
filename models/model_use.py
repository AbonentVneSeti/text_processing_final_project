import importlib
import torch
from .metrics import compute_metrics

def load_model(model_name, model_config, checkpoint_path=None):
    module = importlib.import_module(f"models.{model_name}.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = module.ParaphraserModel(model_config, device)
    if checkpoint_path:
        model.load(checkpoint_path)
    return model

def generate_paraphrases(texts, model):
    return model.generate(texts)

def evaluate_model(model, test_loader, metrics_config):
    predictions = model.generate(test_loader.dataset['original'].tolist())
    references = test_loader.dataset['paraphrase'].tolist()
    metrics = compute_metrics(predictions, references,
                              metrics_config.get('metrics', []),
                              metrics_config)
    return metrics