import torch
import torch.nn as nn


class Additive(nn.Module):
    r"""Bahdanau (additive) attention, from arXiv:1409.0473.

    At every output step the decoder looks at all the source words and
    takes a weighted mix of them. Three steps:

    1. score each source word j for output step i:
           e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)

    2. turn the scores into weights that add up to 1:
           alpha_ij = exp(e_ij) / sum_k exp(e_ik)

    3. take the weighted mix of the source word vectors:
           c_i = sum_j alpha_ij h_j

    s_{i-1} is the decoder's previous state, h_j is the encoder vector
    for source word j (forward and backward joined, so 2*hidden big).
    W_a, U_a and v_a are the weights the model learns.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))   # starts at zero like the original

    def forward(self, state, annotations, summary, mask):
        r"""Give the decoder its context for one step.

        state       : (B, H)      decoder's previous hidden state
        annotations : (B, T, 2H)  one vector per source word
        summary     : not used here, it is only for the fixed vector model
        mask        : (B, T)      True for real words, False for padding

        returns the context (B, 2H) and the weights (B, T)
        """
        scores = torch.tanh(self.W(state).unsqueeze(1) + self.U(annotations)) @ self.v   # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))   # padding gets zero weight
        weights = torch.softmax(scores, dim=-1)                                          # (B, T)
        context = torch.bmm(weights.unsqueeze(1), annotations).squeeze(1)                # (B, 2H)
        return context, weights