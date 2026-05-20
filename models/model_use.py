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

def generate_paraphrases(texts, model, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results.extend(model.generate(batch))
    return results

def evaluate_model(model, test_loader, metrics_config):
    batch_size = metrics_config.get("batch_size", 32)
    predictions = generate_paraphrases(
        test_loader.dataset['original'].tolist(),
        model,
        batch_size
    )
    references = test_loader.dataset['paraphrase'].tolist()
    metrics = compute_metrics(
        predictions, references,
        metrics_config.get('metrics', []),
        metrics_config
    )
    return metrics