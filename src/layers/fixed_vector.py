import torch.nn as nn


class FixedVector(nn.Module):
    # the plain model: same summary vector at every step, no weights
    def forward(self, state, annotations, summary, mask):
        # second value is the attention weights, there are none here
        return summary, None