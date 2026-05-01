import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer

def bleu_score(predictions, references, **kwargs):
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def bert_score(predictions, references, model_name="cointegrated/rubert-tiny2", batch_size=16):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=model_name,
        batch_size=batch_size,
        lang="ru"
    )
    return np.mean(results["f1"])

def cosine_similarity_embeddings(predictions, references, model_name="sentence-transformers/LaBSE", batch_size=16):
    model = SentenceTransformer(model_name)
    emb_pred = model.encode(predictions, batch_size=batch_size)
    emb_ref = model.encode(references, batch_size=batch_size)
    sim = np.sum(emb_pred * emb_ref, axis=1) / (
        np.linalg.norm(emb_pred, axis=1) * np.linalg.norm(emb_ref, axis=1)
    )
    return np.mean(sim)

def compute_metrics(predictions, references, metrics_list, config):
    results = {}
    for metric in metrics_list:
        if metric == "bleu":
            results["bleu"] = bleu_score(predictions, references)
        elif metric == "bertscore":
            model = config.get("bertscore_model", "cointegrated/rubert-tiny2")
            bs = config.get("batch_size", 16)
            results["bertscore"] = bert_score(predictions, references, model, bs)
        elif metric == "cosine_similarity":
            model = config.get("embedding_model", "sentence-transformers/LaBSE")
            bs = config.get("batch_size", 16)
            results["cosine_similarity"] = cosine_similarity_embeddings(predictions, references, model, bs)
    return results