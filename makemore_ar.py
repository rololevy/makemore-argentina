"""
MakeMore Argentina — Character-level language models for Argentine names and surnames.
Adapted from Andrej Karpathy's makemore (https://github.com/karpathy/makemore).

This module contains all model architectures and utility functions, importable
by both the training script and the Streamlit app.
"""

import math
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

# -----------------------------------------------------------------------------
# Model Config

@dataclass
class ModelConfig:
    block_size: int = None   # length of input sequences
    vocab_size: int = None   # input integers in range [0 .. vocab_size - 1]
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (GPT-2 style)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words

class CausalBoW(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x
        return y

class BoWBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, config.n_embd2),
            c_proj=nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.context_block = BoWBlock(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.context_block(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# RNN / GRU

class RNNCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        z = F.sigmoid(self.xh_to_z(xh))
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):
    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        emb = self.wte(idx)
        hprev = self.start.expand((b, -1))
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]
            ht = self.cell(xt, hprev)
            hprev = ht
            hiddens.append(ht)
        hidden = torch.stack(hiddens, 1)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# MLP (Bengio et al. 2003)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)  # +1 for <BLANK>
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size  # <BLANK>
            embs.append(tok_emb)
        x = torch.cat(embs, -1)
        logits = self.mlp(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Bigram

class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1

    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Dataset and generation utilities

class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1  # characters + special 0 token

    def get_output_length(self):
        return self.max_word_length + 1  # <START> token + word

    def encode(self, word):
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix):
        return ''.join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1  # mask loss at inactive locations
        return x, y


def create_datasets(input_file):
    """Load a text file (one word per line) and create train/test CharDatasets."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = sorted(list(set(''.join(words))))
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    return train_dataset, test_dataset


class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
# Generation

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Autoregressively generate characters. idx is (b, t) of starting tokens.
    Returns the full sequence including the initial tokens.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def generate_completions(model, dataset, prefix: str, num_samples: int = 10,
                         temperature: float = 1.0, top_k: int = None):
    """
    Given a prefix string, generate completions using the model.
    Returns list of completed words.
    """
    model.eval()
    max_len = dataset.get_output_length() - 1

    # Encode prefix: start with 0 (<START>) then encode prefix chars
    prefix_tokens = [0]  # <START>
    for ch in prefix:
        if ch in dataset.stoi:
            prefix_tokens.append(dataset.stoi[ch])
        # skip unknown characters

    idx = torch.tensor(prefix_tokens, dtype=torch.long).unsqueeze(0)
    idx = idx.repeat(num_samples, 1)

    remaining = max_len - len(prefix_tokens) + 1
    if remaining <= 0:
        return [prefix] * num_samples

    X_samp = generate(model, idx, remaining, temperature=temperature,
                      do_sample=True, top_k=top_k)

    results = []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()  # crop <START>
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word = dataset.decode(row)
        results.append(word)

    return results


@torch.no_grad()
def generate_names(model, dataset, num_samples: int = 10,
                   temperature: float = 1.0, top_k: int = None):
    """
    Generate names from scratch (no prefix).
    Returns list of (word, category) where category is 'train', 'novel'.
    """
    model.eval()
    X_init = torch.zeros(num_samples, 1, dtype=torch.long)
    steps = dataset.get_output_length() - 1

    X_samp = generate(model, X_init, steps, temperature=temperature,
                      do_sample=True, top_k=top_k)

    results = []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word = dataset.decode(row)
        category = 'existente' if dataset.contains(word) else 'novel'
        results.append((word, category))

    return results


# Model registry
MODEL_CLASSES = {
    'bigram': lambda config: Bigram(config),
    'mlp': lambda config: MLP(config),
    'rnn': lambda config: RNN(config, cell_type='rnn'),
    'gru': lambda config: RNN(config, cell_type='gru'),
    'bow': lambda config: BoW(config),
    'transformer': lambda config: Transformer(config),
}

MODEL_DESCRIPTIONS = {
    'bigram': 'Bigram — tabla de probabilidades de pares de caracteres',
    'mlp': 'MLP — red neuronal feedforward (Bengio et al. 2003)',
    'rnn': 'RNN — red neuronal recurrente vanilla',
    'gru': 'GRU — Gated Recurrent Unit (variante avanzada de RNN)',
    'bow': 'BoW — Bag of Words con atención causal',
    'transformer': 'Transformer — arquitectura GPT-2',
}

MODEL_FRIENDLY_INFO = {
    'bigram': 'El más simple. Mira solo la última letra para decidir la siguiente. Resultados básicos pero rápidos.',
    'mlp': 'Mira las últimas letras juntas para decidir la siguiente. Genera nombres más coherentes que el Bigram.',
    'bow': 'Considera todas las letras anteriores con un sistema de atención. Buenos resultados con bajo costo.',
    'rnn': 'Tiene "memoria" — procesa letra por letra recordando lo anterior. Captura patrones más largos.',
    'gru': 'Versión mejorada del RNN con mejor memoria. Aprende cuándo recordar y cuándo olvidar.',
    'transformer': 'El más avanzado. Usa atención para conectar todas las letras entre sí. La misma tecnología detrás de ChatGPT, adaptada a nombres argentinos.',
}


def create_model(model_type: str, config: ModelConfig):
    """Create a model by type name."""
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_type](config)


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a trained model checkpoint. The checkpoint contains the model state dict,
    model config, model type, dataset chars, and max_word_length.
    Returns (model, dataset).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(
        vocab_size=checkpoint['vocab_size'],
        block_size=checkpoint['block_size'],
        n_layer=checkpoint.get('n_layer', 4),
        n_head=checkpoint.get('n_head', 4),
        n_embd=checkpoint.get('n_embd', 64),
        n_embd2=checkpoint.get('n_embd2', 64),
    )
    model = create_model(checkpoint['model_type'], config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Reconstruct dataset for encoding/decoding
    chars = checkpoint['chars']
    max_word_length = checkpoint['max_word_length']

    # Load training words from the dataset file for contains() checks
    train_words = checkpoint.get('train_words', [])
    if not train_words:
        dataset_file = checkpoint.get('dataset_file', '')
        if dataset_file:
            # Look for the dataset file relative to the checkpoint
            data_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'data')
            # Also check in same directory level
            for search_dir in [data_dir, os.path.join(os.path.dirname(checkpoint_path), '..', 'data')]:
                candidate = os.path.join(search_dir, dataset_file)
                if os.path.exists(candidate):
                    with open(candidate, 'r', encoding='utf-8') as f:
                        train_words = [w.strip() for w in f if w.strip()]
                    break

    dataset = CharDataset(train_words, chars, max_word_length)
    return model, dataset
