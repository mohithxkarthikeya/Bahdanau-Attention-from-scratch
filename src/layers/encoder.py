import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data import PAD


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD)
        self.dropout = nn.Dropout(dropout)
        # bidirectional: one GRU reads left to right, the other right to left
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True, bidirectional=True)
        # forward + backward summary is 2*hidden, the decoder wants hidden
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src, lengths):
        batch_size, max_len = src.shape
        embedded = self.dropout(self.embedding(src))

        # packing makes the GRU skip the padding
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed)
        annotations, _ = pad_packed_sequence(packed_out, batch_first=True,
                                             total_length=max_len)   # (B, T, 2H)

        # h[0] = forward last state, h[1] = backward last state
        summary = torch.cat([h[0], h[1]], dim=-1)                    # (B, 2H)
        start_state = torch.tanh(self.bridge(summary))               # (B, H)

        positions = torch.arange(max_len, device=src.device)
        mask = positions[None, :] < lengths[:, None]                 # True = real word
        return annotations, mask, summary, start_state