import os
import re
import json
import pickle
import random
from collections import Counter
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np


class WordTokenizer:
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

    def __init__(self, vocab_size: int = 30_000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+|[^\s\w]", text.lower())

    def build_vocab(self, sentences: List[str]):
        counter: Counter = Counter()
        for s in sentences:
            counter.update(self._tokenize(s))

        specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        most_common = [w for w, _ in counter.most_common(self.vocab_size - len(specials))]
        vocab = specials + most_common

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        if add_bos:
            ids = [self.word2idx[self.BOS]] + ids
        if add_eos:
            ids = ids + [self.word2idx[self.EOS]]
        return ids

    def decode(self, ids: List[int]) -> str:
        specials = {self.word2idx[s] for s in [self.PAD, self.BOS, self.EOS]}
        words = [self.idx2word.get(i, self.UNK) for i in ids if i not in specials]
        result = ""
        for w in words:
            if result and re.match(r"[^\s\w]", w):
                result += w
            else:
                result += (" " if result else "") + w
        return result.strip()

    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]

    def bos_idx(self) -> int:
        return self.word2idx[self.BOS]

    def eos_idx(self) -> int:
        return self.word2idx[self.EOS]

    def save(self, path: str):
        with open(os.path.join(path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        with open(os.path.join(path, "tokenizer.pkl"), "rb") as f:
            self.__dict__.update(pickle.load(f))


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_keys  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v       = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, mask=None):
        scores = self.v(torch.tanh(
            self.W_query(query).unsqueeze(1) + self.W_keys(keys)
        )).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return context, weights


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = (outputs[:, :, :outputs.size(-1)//2] +
                   outputs[:, :, outputs.size(-1)//2:])
        hidden = torch.tanh(self.proj(
            torch.cat([hidden[-2], hidden[-1]], dim=-1)
        ))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention  = BahdanauAttention(hidden_size)
        self.rnn = nn.GRU(
            embed_dim + hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, tgt_token, hidden, enc_outputs, src_mask):
        emb = self.dropout(self.embedding(tgt_token))
        context, _ = self.attention(hidden[-1], enc_outputs, src_mask)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.out(torch.cat([output.squeeze(1), context], dim=-1))
        return logits, hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        B, T_tgt = tgt.shape
        vocab_size = self.decoder.out.out_features

        enc_outputs, hidden = self.encoder(src, src_len)
        src_mask = (src == self.encoder.embedding.padding_idx)

        num_dec_layers = self.decoder.rnn.num_layers
        hidden = hidden.unsqueeze(0).repeat(num_dec_layers, 1, 1)

        outputs = torch.zeros(B, T_tgt, vocab_size, device=src.device)
        inp = tgt[:, 0:1]

        for t in range(1, T_tgt):
            logits, hidden = self.decoder.forward_step(inp, hidden, enc_outputs, src_mask)
            outputs[:, t] = logits
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            inp = tgt[:, t:t+1] if teacher_force else logits.argmax(-1).unsqueeze(1)

        return outputs


class ParaphraserModel:
    def __init__(self, model_config: dict, device: str = None):
        self.config      = model_config
        self.vocab_size  = model_config.get("vocab_size", 30_000)
        self.embed_dim   = model_config.get("embed_dim", 128)
        self.hidden_size = model_config.get("hidden_size", 256)
        self.num_layers  = model_config.get("num_layers", 2)
        self.dropout     = model_config.get("dropout", 0.3)
        self.max_length  = model_config.get("max_length", 80)
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = WordTokenizer(self.vocab_size)
        self.model: Optional[Seq2SeqModel] = None
        self.optimizer = None
        self.scheduler = None

    def _build_model(self):
        vocab_size = len(self.tokenizer.word2idx)
        pad_idx    = self.tokenizer.pad_idx()
        encoder = Encoder(vocab_size, self.embed_dim, self.hidden_size,
                          self.num_layers, self.dropout, pad_idx)
        decoder = Decoder(vocab_size, self.embed_dim, self.hidden_size,
                          self.num_layers, self.dropout, pad_idx)
        self.model = Seq2SeqModel(encoder, decoder).to(self.device)

    def _collate(self, originals: List[str], paraphrases: List[str] | None = None):
        src_ids  = [self.tokenizer.encode(s, add_bos=False, add_eos=True)[:self.max_length]
                    for s in originals]
        src_lens = torch.tensor([len(s) for s in src_ids])
        max_src  = src_lens.max().item()
        pad = self.tokenizer.pad_idx()

        src_tensor = torch.tensor(
            [s + [pad] * (max_src - len(s)) for s in src_ids],
            dtype=torch.long, device=self.device
        )

        if paraphrases is None:
            return src_tensor, src_lens

        tgt_ids  = [self.tokenizer.encode(t, add_bos=True, add_eos=True)[:self.max_length]
                    for t in paraphrases]
        max_tgt  = max(len(t) for t in tgt_ids)
        tgt_tensor = torch.tensor(
            [t + [pad] * (max_tgt - len(t)) for t in tgt_ids],
            dtype=torch.long, device=self.device
        )
        return src_tensor, src_lens, tgt_tensor

    def _save_checkpoint(self, checkpoint_dir: str, epoch: int,
                         best_val_loss: float, history: List[Dict[str, Any]]):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.tokenizer.save(checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model_state.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer_state.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler_state.pt"))

        training_state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f)

        torch.save(torch.get_rng_state(), os.path.join(checkpoint_dir, "rng_state_torch.pt"))
        # numpy random state saved via pickle due to inhomogeneous structure
        with open(os.path.join(checkpoint_dir, "rng_state_numpy.pkl"), "wb") as f:
            pickle.dump(np.random.get_state(), f)
        with open(os.path.join(checkpoint_dir, "rng_state_random.pkl"), "wb") as f:
            pickle.dump(random.getstate(), f)

        arch = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_length": self.max_length,
        }
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(arch, f)

    def _load_checkpoint(self, checkpoint_dir: str):
        self.tokenizer.load(checkpoint_dir)
        self._build_model()
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model_state.pt"), map_location=self.device)
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer_state.pt"), map_location=self.device)
        )
        self.scheduler.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "scheduler_state.pt"), map_location=self.device)
        )

        with open(os.path.join(checkpoint_dir, "training_state.json")) as f:
            training_state = json.load(f)
        torch.set_rng_state(torch.load(os.path.join(checkpoint_dir, "rng_state_torch.pt")))
        with open(os.path.join(checkpoint_dir, "rng_state_numpy.pkl"), "rb") as f:
            np.random.set_state(pickle.load(f))
        with open(os.path.join(checkpoint_dir, "rng_state_random.pkl"), "rb") as f:
            random.setstate(pickle.load(f))

        return training_state

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        if not os.path.isdir(output_dir):
            return None
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            return None
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        return os.path.join(output_dir, latest)

    def _cleanup_checkpoints(self, output_dir: str, save_total_limit: int):
        if save_total_limit <= 0:
            return
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) <= save_total_limit:
            return
        sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        for ckpt in sorted_checkpoints[:-save_total_limit]:
            full_path = os.path.join(output_dir, ckpt)
            if os.path.isdir(full_path):
                for f in os.listdir(full_path):
                    os.remove(os.path.join(full_path, f))
                os.rmdir(full_path)

    def train(self, train_loader, val_loader, trainer_config=None, metrics_config=None):
        trainer_config = trainer_config or {}
        lr          = float(self.config.get("learning_rate", 1e-3))
        num_epochs  = int(self.config.get("num_epochs", 3))
        batch_size  = int(self.config.get("batch_size", 64))
        output_dir  = trainer_config.get("output_dir", "./saves")
        os.makedirs(output_dir, exist_ok=True)

        save_strategy = trainer_config.get("save_strategy", "epoch")
        save_total_limit = int(trainer_config.get("save_total_limit", 3))
        resume_from_checkpoint = trainer_config.get("resume_from_checkpoint", True)
        load_best_model_at_end = trainer_config.get("load_best_model_at_end", False)
        metric_for_best_model = trainer_config.get("metric_for_best_model", "eval_loss")
        greater_is_better = trainer_config.get("greater_is_better", False)

        start_epoch = 1
        best_val_loss = float("inf")
        history = []

        ckpt_dir = self._find_latest_checkpoint(output_dir) if resume_from_checkpoint else None
        if ckpt_dir is not None:
            training_state = self._load_checkpoint(ckpt_dir)
            start_epoch = training_state["epoch"] + 1
            best_val_loss = training_state["best_val_loss"]
            history = training_state["history"]
            self.optimizer.param_groups[0]['lr'] = lr
            print(f"Resumed from checkpoint {ckpt_dir}, epoch {start_epoch}")
        else:
            all_sents: List[str] = []
            for batch_df in tqdm(train_loader, desc="Reading dataset"):
                all_sents.extend(batch_df["original"].tolist())
                all_sents.extend(batch_df["paraphrase"].tolist())
            self.tokenizer.build_vocab(all_sents)
            self._build_model()
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=1, factor=0.5, verbose=True)

        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_idx())

        for epoch in range(start_epoch, num_epochs + 1):
            self.model.train()
            total_loss, steps = 0.0, 0
            tf_ratio = max(0.3, 1.0 - (epoch - 1) * 0.2)

            batches = list(train_loader)
            pbar = tqdm(batches, desc=f"Epoch {epoch}/{num_epochs} [train]")
            for batch_df in pbar:
                orig_batch = batch_df["original"].tolist()
                para_batch = batch_df["paraphrase"].tolist()
                for i in range(0, len(orig_batch), batch_size):
                    src_b = orig_batch[i:i+batch_size]
                    tgt_b = para_batch[i:i+batch_size]
                    if not src_b:
                        continue
                    src, src_lens, tgt = self._collate(src_b, tgt_b)
                    self.optimizer.zero_grad()
                    logits = self.model(src, src_lens, tgt, tf_ratio)
                    loss = criterion(
                        logits[:, 1:].reshape(-1, logits.size(-1)),
                        tgt[:, 1:].reshape(-1)
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                    steps += 1
                    pbar.set_postfix({"loss": f"{total_loss/steps:.4f}", "tf": f"{tf_ratio:.2f}"})

            avg_train = total_loss / max(steps, 1)

            self.model.eval()
            val_loss, val_steps = 0.0, 0
            with torch.no_grad():
                for batch_df in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]"):
                    orig_batch = batch_df["original"].tolist()
                    para_batch = batch_df["paraphrase"].tolist()
                    for i in range(0, len(orig_batch), batch_size):
                        src_b = orig_batch[i:i+batch_size]
                        tgt_b = para_batch[i:i+batch_size]
                        if not src_b:
                            continue
                        src, src_lens, tgt = self._collate(src_b, tgt_b)
                        logits = self.model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
                        loss = criterion(
                            logits[:, 1:].reshape(-1, logits.size(-1)),
                            tgt[:, 1:].reshape(-1)
                        )
                        val_loss += loss.item()
                        val_steps += 1

            avg_val = val_loss / max(val_steps, 1)
            self.scheduler.step(avg_val)

            record = {"epoch": epoch, "loss": avg_train, "eval_loss": avg_val}

            if metrics_config and metrics_config.get("metrics"):
                from ..metrics import compute_metrics
                val_df = val_loader.dataset
                sample_size = min(500, len(val_df))
                val_sample = val_df.sample(n=sample_size, random_state=42)
                originals = val_sample["original"].tolist()
                references = val_sample["paraphrase"].tolist()
                predictions = []
                for orig in originals:
                    pred = self.generate([orig])[0]
                    predictions.append(pred)
                mets = compute_metrics(predictions, references, metrics_config['metrics'], metrics_config)
                for k, v in mets.items():
                    record[f"val_{k}"] = float(v)
                print(f"Metrics: { {k: f'{v:.4f}' for k, v in mets.items()} }")

            history.append(record)
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

            if save_strategy == "epoch":
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
                self._save_checkpoint(checkpoint_dir, epoch, best_val_loss, history)
                self._cleanup_checkpoints(output_dir, save_total_limit)

            if avg_val < best_val_loss:
                best_val_loss = avg_val

        if load_best_model_at_end:
            best_ckpt = None
            best_value = float("inf") if not greater_is_better else -float("inf")
            for d in os.listdir(output_dir):
                if d.startswith("checkpoint-"):
                    with open(os.path.join(output_dir, d, "training_state.json")) as f:
                        state = json.load(f)
                    for entry in state["history"]:
                        if entry.get("epoch") == int(d.split("-")[1]):
                            val_metric = entry.get(metric_for_best_model)
                            if val_metric is not None:
                                if (greater_is_better and val_metric > best_value) or \
                                   (not greater_is_better and val_metric < best_value):
                                    best_value = val_metric
                                    best_ckpt = os.path.join(output_dir, d)
            if best_ckpt is not None:
                self._load_checkpoint(best_ckpt)
                self.save(output_dir)
                print(f"Loaded best model from {best_ckpt}")

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            serializable = []
            for rec in history:
                clean_rec = {}
                for k, v in rec.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean_rec[k] = float(v)
                    else:
                        clean_rec[k] = v
                serializable.append(clean_rec)
            json.dump(serializable, f)

        return history

    def generate(self, texts: List[str], num_return_sequences: int = 1) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        self.model.eval()

        results = []
        pad = self.tokenizer.pad_idx()
        bos = self.tokenizer.bos_idx()
        eos = self.tokenizer.eos_idx()

        with torch.no_grad():
            for text in texts:
                src_ids  = self.tokenizer.encode(text, add_bos=False, add_eos=True)[:self.max_length]
                src_len  = torch.tensor([len(src_ids)])
                src      = torch.tensor([src_ids], dtype=torch.long, device=self.device)

                enc_out, hidden = self.model.encoder(src, src_len)
                src_mask = (src == pad)
                num_dec_layers = self.model.decoder.rnn.num_layers
                hidden = hidden.unsqueeze(0).repeat(num_dec_layers, 1, 1)

                inp = torch.tensor([[bos]], dtype=torch.long, device=self.device)
                generated = []

                for _ in range(self.max_length):
                    logits, hidden = self.model.decoder.forward_step(
                        inp, hidden, enc_out, src_mask
                    )
                    next_token = logits.argmax(-1).item()
                    if next_token == eos:
                        break
                    generated.append(next_token)
                    inp = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

                results.append(self.tokenizer.decode(generated))

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save(path)
        arch = {
            "vocab_size":  self.vocab_size,
            "embed_dim":   self.embed_dim,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     self.dropout,
            "max_length":  self.max_length,
        }
        with open(os.path.join(path, "arch.json"), "w") as f:
            json.dump(arch, f)

    def load(self, path: str):
        with open(os.path.join(path, "arch.json")) as f:
            arch = json.load(f)
        self.vocab_size  = arch["vocab_size"]
        self.embed_dim   = arch["embed_dim"]
        self.hidden_size = arch["hidden_size"]
        self.num_layers  = arch["num_layers"]
        self.dropout     = arch["dropout"]
        self.max_length  = arch["max_length"]

        self.tokenizer.load(path)
        self._build_model()
        state = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()