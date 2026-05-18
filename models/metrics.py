import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

_bert_model = None
_bert_tokenizer = None
_sim_model = None

def _load_bert(model_name):
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bert_model = AutoModel.from_pretrained(model_name).to(device)
        _bert_model.eval()
    return _bert_model, _bert_tokenizer

def _load_sim_model(model_name):
    global _sim_model
    if _sim_model is None:
        _sim_model = SentenceTransformer(model_name)
    return _sim_model

def bleu_score(predictions, references):
    bleu = evaluate.load("sacrebleu")
    results = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def custom_bertscore(predictions, references, model_name="cointegrated/rubert-tiny2", batch_size=16):
    model, tokenizer = _load_bert(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_f1 = []
    with torch.no_grad():
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i+batch_size]
            batch_refs = references[i:i+batch_size]
            inputs_pred = tokenizer(batch_preds, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs_ref = tokenizer(batch_refs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs_pred = {k: v.to(device) for k, v in inputs_pred.items()}
            inputs_ref = {k: v.to(device) for k, v in inputs_ref.items()}
            outputs_pred = model(**inputs_pred).last_hidden_state
            outputs_ref = model(**inputs_ref).last_hidden_state
            for j in range(len(batch_preds)):
                mask_pred = inputs_pred["attention_mask"][j].bool()
                mask_ref = inputs_ref["attention_mask"][j].bool()
                emb_pred = outputs_pred[j][mask_pred]
                emb_ref = outputs_ref[j][mask_ref]
                emb_pred = torch.nn.functional.normalize(emb_pred, p=2, dim=1)
                emb_ref = torch.nn.functional.normalize(emb_ref, p=2, dim=1)
                sim = torch.mm(emb_pred, emb_ref.t())
                precision = sim.max(dim=1)[0].mean().item()
                recall = sim.max(dim=0)[0].mean().item()
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                all_f1.append(f1)
    return np.mean(all_f1)

def cosine_similarity_embeddings(predictions, references, model_name="sentence-transformers/LaBSE", batch_size=16):
    model = _load_sim_model(model_name)
    emb_pred = model.encode(predictions, batch_size=batch_size, show_progress_bar=False)
    emb_ref = model.encode(references, batch_size=batch_size, show_progress_bar=False)
    sim = np.sum(emb_pred * emb_ref, axis=1) / (
        np.linalg.norm(emb_pred, axis=1) * np.linalg.norm(emb_ref, axis=1)
    )
    return np.mean(sim)

def compute_metrics_for_trainer(tokenizer, metrics_list, config):
    def compute_metrics(eval_preds):
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.array(predictions).astype(int)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = labels.astype(int)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        results = {}
        for metric in metrics_list:
            if metric == "bleu":
                results["bleu"] = bleu_score(decoded_preds, decoded_labels)
            elif metric == "bertscore":
                model = config.get("bertscore_model", "cointegrated/rubert-tiny2")
                bs = config.get("batch_size", 16)
                results["bertscore"] = custom_bertscore(decoded_preds, decoded_labels, model, bs)
            elif metric == "cosine_similarity":
                model = config.get("embedding_model", "sentence-transformers/LaBSE")
                bs = config.get("batch_size", 16)
                results["cosine_similarity"] = cosine_similarity_embeddings(decoded_preds, decoded_labels, model, bs)
        return results
    return compute_metrics