import torch
import torch.nn as nn

from src.data import PAD, SOS, EOS
from src.layers import get_layer
from src.layers.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, attention, src_vocab_size, tgt_vocab_size,
                 emb_size=256, hidden_size=256, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_size, hidden_size, dropout)
        # either FixedVector or Additive, chosen by name
        self.attention = get_layer(attention, hidden_size)

        self.embedding = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        # input = previous word + context (context is 2*hidden)
        self.cell = nn.GRUCell(emb_size + 2 * hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size + 2 * hidden_size + emb_size, tgt_vocab_size)

    def step(self, token, state, annotations, summary, mask):
        emb = self.dropout(self.embedding(token))                          # (B, emb)
        context, weights = self.attention(state, annotations, summary, mask)
        state = self.cell(torch.cat([emb, context], dim=-1), state)        # (B, H)
        logits = self.output(self.dropout(torch.cat([state, context, emb], dim=-1)))
        return logits, state, weights

    def forward(self, src, lengths, tgt):
        # teacher forcing: the decoder always gets the true previous word
        annotations, mask, summary, state = self.encoder(src, lengths)
        all_logits = []
        for t in range(tgt.size(1) - 1):
            logits, state, _ = self.step(tgt[:, t], state, annotations, summary, mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)            # (B, T_tgt - 1, vocab)

    @torch.no_grad()
    def decode(self, src, lengths, max_len=50):
        # greedy: always take the most likely next word
        self.eval()
        annotations, mask, summary, state = self.encoder(src, lengths)
        batch_size = src.size(0)
        token = torch.full((batch_size,), SOS, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        outputs, all_weights = [], []
        for _ in range(max_len):
            logits, state, weights = self.step(token, state, annotations, summary, mask)
            token = logits.argmax(dim=-1).masked_fill(finished, PAD)
            outputs.append(token)
            all_weights.append(weights)
            finished = finished | (token == EOS)
            if finished.all():
                break

        outputs = torch.stack(outputs, dim=1)
        if all_weights[0] is None:
            return outputs, None
        return outputs, torch.stack(all_weights, dim=1)   # (B, steps, T)