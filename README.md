# Bahdanau-Attention-from-scratch

A small PyTorch re-implementation of [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014) paper, built to see myself that how attention is better than compressing a whole sentence into one vector.

"From scratch" means, I wrote the whole encoder, additive (attention) layer and the decoder myself. I use PyTorch's basic pieces (GRU, Embeddings, Linear Layers) and no seq2seq library.

## The paper in short 

Paper: Bahdanau, Cho and Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014). arXiv 1409.0473.

### The problem before this paper

Translation models had two parts, one is encoder which read the sentence and compressed a whole sentence into one fixed length vector and a decoder wrote the translation using that encoder vector.

That is a huge bottleneck. The vector is the same size for a 5 words sentence or a 50 words sentence, so a long sentence has to be compressed into same small space. The decoder cannot go back and look at one particular source word while it writes.

In this type of models, the performance deteriorates rapidly as the length of an input sentence increases.[2]

### What is solves

- The single-vector bottleneck is gone, because the decoder can use any source word at any step.
- Different output words can look at different source words, which helps when the word order differs between the two languages.
- The weights show what the model looked at for each output word, so the alignments can be plotted and checked by eye. The paper says the learned alignments matched intuition.
- On English to French, the paper reports translation quality comparable to the phrase-based systems of that time.

### What this repo does

This repo does not try to reproduce the paper's English to French results. It re-implements the idea from scratch and compares attention against the plain fixed-vector model on a small dataset (Multi30k, English to German).

## Data

Multi30k, English to German. It is a set of short image captions with their German translations. I use the text-only version (`bentrevett/multi30k`), so there are no images.

- 29,000 training pairs
- 1,014 validation pairs (for tuning and early stopping)
- 1,000 test pairs (used once at the end)


## The two models

Both use the same bidirectional GRU encoder, the same sizes and the same decoder start state. Only the context given to the decoder is different.

- `fixed_vector`: the decoder gets the same summary vector at every step.
- `additive`: at every step the decoder scores all source words, turns the scores into weights, and takes a weighted mix of the words (Bahdanau attention)[1].

### How the additive layer works

"Additive" is the way the score is made: the decoder state and each source word vector are projected, added together, passed through tanh, and turned into one number. There are three steps, and they are the three formulas of the paper ([arXiv:1409.0473](https://arxiv.org/abs/1409.0473)).

**1. Score.** How useful is source word `j` for the word I am about to write (output step `i`)?

$$
e_{ij} = v_a^{\top} \tanh\left(W_a \, s_{i-1} + U_a \, h_j\right)
$$

**2. Weights.** Turn the scores into weights that add up to 1.

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

**3. Context.** Take the weighted mix of the source word vectors.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} \, h_j
$$

What the letters mean:

- `s_{i-1}`: the decoder's previous hidden state
- `h_j`: the encoder vector for source word `j` (forward and backward states joined)
- `W_a`, `U_a`, `v_a`: weights the model learns, trained together with everything else
- `T_x`: the number of words in the source sentence

In the code, these are in [`src/layers/additive.py`](): step 1 is the `tanh(...) @ v` line, step 2 is the softmax, and step 3 is the `bmm`. One thing the paper doesn't cover: the padding positions get a score of minus infinity before the softmax, so they end up with weight 0.

## How to run

Run everything from the repo root.

```
git clone https://github.com/mohithxkarthikeya/Bahdanau-Attention-from-scratch.git
cd Bahdanau-Attention-from-scratch
pip install -r requirements.txt
python -m pytest
python -m src.train --attention fixed_vector --seed 0
python -m src.train --attention additive --seed 0
python -m src.evaluate --attention fixed_vector --seed 0
python -m src.evaluate --attention additive --seed 0
python -m src.plots
```

Repeat the train and evaluate commands with `--seed 1` and `--seed 2` to get 3 seeds per model. A GPU helps a lot, because the decoder runs one word at a time.

## Files

- `src/data.py`: loads Multi30k, builds the vocabularies, makes batches
- `src/layers/encoder.py`: the bidirectional GRU encoder
- `src/layers/fixed_vector.py`: the plain model (same vector every step)
- `src/layers/additive.py`: Bahdanau attention
- `src/models.py`: the seq2seq model that uses one of the two layers
- `src/train.py`: training with early stopping on validation loss
- `src/evaluate.py`: test loss, BLEU, and BLEU by sentence length
- `src/plots.py`: BLEU vs length plot and attention heatmaps
- `tests/`: shape and mask checks for the encoder and the two layers

## Results

Not run yet. I will add the test BLEU table, `plots/bleu_vs_length.png` and the attention heatmaps here.

## What I found

To be written after the runs, including what I am not sure about.

## Notes

BLEU here is computed on lowercased, word-split text with an unknown-word token, so it is not comparable with numbers from papers. The right comparison is the two models against each other, on the same data.

## Credits

See [CREDITS.md](CREDITS.md).

## References

1. D. Bahdanau, K. Cho, Y. Bengio. [*Neural Machine Translation by Jointly Learning to Align and Translate.* 2014 (ICLR 2015).](https://arxiv.org/abs/1409.0473)
   The paper this repo re-implements: bidirectional encoder and additive attention.

2. K. Cho, B. van Merrienboer, D. Bahdanau, Y. Bengio. [*On the Properties of Neural Machine Translation: Encoder-Decoder Approaches.* 2014.](https://arxiv.org/abs/1409.1259)
   Earlier work on the plain fixed-vector encoder-decoder, including how it behaves on longer sentences.

3. K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio. [*Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* 2014.](https://arxiv.org/abs/1406.1078)
   Introduced the GRU, which the encoder and decoder use here.

4. I. Sutskever, O. Vinyals, Q. V. Le. [*Sequence to Sequence Learning with Neural Networks.* 2014.](https://arxiv.org/abs/1409.3215)
   The encoder-decoder idea that the fixed-vector baseline follows.

5. D. Elliott, S. Frank, K. Sima'an, L. Specia. [*Multi30K: Multilingual English-German Image Descriptions.* 2016.](https://arxiv.org/abs/1605.00459)
   The dataset. This repo uses only the text, from the Hugging Face copy `bentrevett/multi30k`.

6. P. Young, A. Lai, M. Hodosh, J. Hockenmaier. [*From image descriptions to visual denotations.* TACL, 2014.](https://aclanthology.org/Q14-1006/)
   Flickr30k, the image collection whose captions Multi30k is built on.

7. K. Papineni, S. Roukos, T. Ward, W.-J. Zhu. [*BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL, 2002.](https://aclanthology.org/P02-1040/)
   The BLEU score used in evaluation.

8. M. Post. [*A Call for Clarity in Reporting BLEU Scores.* 2018.](https://arxiv.org/abs/1804.08771)
   The `sacrebleu` library used to compute BLEU.

9. A. Paszke et al. [*PyTorch: An Imperative Style, High-Performance Deep Learning Library.* 2019.](https://arxiv.org/abs/1912.01703)
   The framework everything is written in.