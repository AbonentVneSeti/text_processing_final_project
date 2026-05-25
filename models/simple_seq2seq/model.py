import os
import re
import json
import pickle
from collections import Counter
from typing import List

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
        emb = self.dropout(self.embedding(tgt_token))    # (B, 1, E)
        context, _ = self.attention(hidden[-1], enc_outputs, src_mask)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # (B, 1, E+H)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.out(torch.cat([output.squeeze(1), context], dim=-1))  # (B, V)
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
        self.model: Seq2SeqModel | None = None


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

    def train(self, train_loader, val_loader, trainer_config: dict = None, metrics_config: dict = None):
        trainer_config = trainer_config or {}
        lr          = float(self.config.get("learning_rate", 1e-3))
        num_epochs  = int(self.config.get("num_epochs", 3))
        batch_size  = int(self.config.get("batch_size", 64))
        output_dir  = trainer_config.get("output_dir", "./saves")
        os.makedirs(output_dir, exist_ok=True)

        print("Строим словарь из обучающей выборки...")
        all_sents: List[str] = []
        for batch_df in tqdm(train_loader, desc="Чтение датасета"):
            all_sents.extend(batch_df["original"].tolist())
            all_sents.extend(batch_df["paraphrase"].tolist())
        self.tokenizer.build_vocab(all_sents)
        print(f"Словарь: {len(self.tokenizer.word2idx):,} токенов")

        self._build_model()
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Параметров модели: {n_params:,}")

        optimizer  = Adam(self.model.parameters(), lr=lr)
        scheduler  = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, verbose=True)
        criterion  = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_idx())

        history = []

        for epoch in range(1, num_epochs + 1):
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
                    optimizer.zero_grad()
                    logits = self.model(src, src_lens, tgt, tf_ratio)
                    loss = criterion(
                        logits[:, 1:].reshape(-1, logits.size(-1)),
                        tgt[:, 1:].reshape(-1)
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    steps      += 1
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
                        val_loss  += loss.item()
                        val_steps += 1

            avg_val = val_loss / max(val_steps, 1)
            scheduler.step(avg_val)

            record = {"epoch": epoch, "loss": avg_train, "eval_loss": avg_val}
            history.append(record)
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
            self.save(output_dir)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history, f)
        print("История сохранена в", os.path.join(output_dir, "history.json"))
        return history

    def generate(self, texts: List[str], num_return_sequences: int = 1) -> List[str]:
        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите train() или load().")
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
        print(f"Базовая модель загружена из {path}")