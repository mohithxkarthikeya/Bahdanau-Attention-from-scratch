import re
import random
from collections import Counter

import torch
from datasets import load_dataset

PAD, SOS, EOS, UNK = 0, 1, 2, 3


def tokenize(text):
    # lowercase, then split into words and punctuation marks
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, words):
        # unknown words become <unk>
        return [self.stoi.get(w, UNK) for w in words]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == EOS or i == PAD:
                break
            if i == SOS:
                continue
            words.append(self.itos[i])
        return words


def build_vocab(sentences, min_freq=2):
    counts = Counter(word for sent in sentences for word in sent)
    itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
    for word, c in counts.most_common():
        if c >= min_freq:
            itos.append(word)
    return Vocab(itos)


def load_multi30k(src_lang="en", tgt_lang="de"):
    raw = load_dataset("bentrevett/multi30k")
    data = {}
    for split in ["train", "validation", "test"]:
        src = [tokenize(x) for x in raw[split][src_lang]]
        tgt = [tokenize(x) for x in raw[split][tgt_lang]]
        data[split] = (src, tgt)

    # vocab comes from the training set only
    src_vocab = build_vocab(data["train"][0])
    tgt_vocab = build_vocab(data["train"][1])
    return data, src_vocab, tgt_vocab


def pad(seqs):
    longest = max(len(s) for s in seqs)
    out = torch.full((len(seqs), longest), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def make_batches(src_sents, tgt_sents, src_vocab, tgt_vocab, batch_size, shuffle, device):
    order = list(range(len(src_sents)))
    if shuffle:
        random.shuffle(order)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        src_ids = [src_vocab.encode(src_sents[i]) for i in idx]
        tgt_ids = [[SOS] + tgt_vocab.encode(tgt_sents[i]) + [EOS] for i in idx]
        lengths = torch.tensor([len(s) for s in src_ids])
        # idx is returned so we can put translations back in the original order
        yield pad(src_ids).to(device), lengths.to(device), pad(tgt_ids).to(device), idx