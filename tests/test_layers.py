import pytest
import torch

from src.layers import get_layer
from src.model import Seq2Seq

HIDDEN = 4


def fake_inputs():
    torch.manual_seed(0)
    state = torch.randn(2, HIDDEN)
    annotations = torch.randn(2, 5, 2 * HIDDEN)
    summary = torch.randn(2, 2 * HIDDEN)
    # second sentence has only 3 real words
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    return state, annotations, summary, mask


def fake_batch():
    src = torch.tensor([[4, 5, 6, 7, 8],
                        [9, 10, 11, 0, 0]])
    lengths = torch.tensor([5, 3])
    tgt = torch.tensor([[1, 5, 6, 7, 2],
                        [1, 8, 9, 2, 0]])
    return src, lengths, tgt


def test_fixed_vector_gives_summary_and_no_weights():
    state, annotations, summary, mask = fake_inputs()
    context, weights = get_layer("fixed_vector", HIDDEN)(state, annotations, summary, mask)
    assert torch.equal(context, summary)
    assert weights is None


def test_additive_shapes():
    state, annotations, summary, mask = fake_inputs()
    context, weights = get_layer("additive", HIDDEN)(state, annotations, summary, mask)
    assert context.shape == (2, 2 * HIDDEN)
    assert weights.shape == (2, 5)


def test_additive_weights_sum_to_one_and_ignore_padding():
    state, annotations, summary, mask = fake_inputs()
    layer = get_layer("additive", HIDDEN)
    layer.v.data = torch.randn(HIDDEN)   # v starts at zero, give it real values for the test
    _, weights = layer(state, annotations, summary, mask)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(weights[1, 3:] == 0)


def test_unknown_layer_name():
    with pytest.raises(ValueError):
        get_layer("something_else", HIDDEN)


def test_both_models_give_the_right_shape():
    src, lengths, tgt = fake_batch()
    for name in ["fixed_vector", "additive"]:
        model = Seq2Seq(name, 20, 25, emb_size=6, hidden_size=HIDDEN, dropout=0.0)
        logits = model(src, lengths, tgt)
        assert logits.shape == (2, tgt.size(1) - 1, 25)


def test_same_seed_same_result():
    src, lengths, tgt = fake_batch()
    outputs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = Seq2Seq("additive", 20, 25, emb_size=6, hidden_size=HIDDEN, dropout=0.0)
        model.eval()
        outputs.append(model(src, lengths, tgt))
    assert torch.allclose(outputs[0], outputs[1])