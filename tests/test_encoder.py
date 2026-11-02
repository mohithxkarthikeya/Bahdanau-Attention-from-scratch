import torch

from src.layers.encoder import Encoder

HIDDEN = 4


def make_encoder():
    torch.manual_seed(0)
    encoder = Encoder(vocab_size=20, emb_size=6, hidden_size=HIDDEN, dropout=0.0)
    encoder.eval()
    return encoder


def make_inputs():
    # sentence 1 has 5 real words, sentence 2 has 3 real words and 2 padding
    src = torch.tensor([[4, 5, 6, 7, 8],
                        [9, 10, 11, 0, 0]])
    lengths = torch.tensor([5, 3])
    return src, lengths


def test_shapes():
    src, lengths = make_inputs()
    annotations, mask, summary, start_state = make_encoder()(src, lengths)
    assert annotations.shape == (2, 5, 2 * HIDDEN)
    assert mask.shape == (2, 5)
    assert summary.shape == (2, 2 * HIDDEN)
    assert start_state.shape == (2, HIDDEN)


def test_mask():
    src, lengths = make_inputs()
    _, mask, _, _ = make_encoder()(src, lengths)
    assert mask[0].tolist() == [True] * 5
    assert mask[1].tolist() == [True, True, True, False, False]


def test_padding_positions_are_zero():
    src, lengths = make_inputs()
    annotations, _, _, _ = make_encoder()(src, lengths)
    assert torch.all(annotations[1, 3:] == 0)


def test_summary_matches_annotations():
    # the forward GRU ends at the last real word,
    # the backward GRU ends at the first word
    src, lengths = make_inputs()
    annotations, _, summary, _ = make_encoder()(src, lengths)
    for b in range(2):
        n = int(lengths[b])
        assert torch.allclose(summary[b, :HIDDEN], annotations[b, n - 1, :HIDDEN], atol=1e-6)
        assert torch.allclose(summary[b, HIDDEN:], annotations[b, 0, HIDDEN:], atol=1e-6)


def test_padding_does_not_change_result():
    # the short sentence alone should give the same numbers as inside a padded batch
    encoder = make_encoder()
    src, lengths = make_inputs()
    annotations, _, _, _ = encoder(src, lengths)
    alone, _, _, _ = encoder(src[1:2, :3], lengths[1:2])
    assert torch.allclose(alone[0], annotations[1, :3], atol=1e-5)