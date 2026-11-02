import argparse
import csv
import math
import os

import sacrebleu
import torch

from src.data import load_multi30k, make_batches, Vocab
from src.model import Seq2Seq
from src.train import average_loss, get_device

# groups of source sentence length (in words) for the BLEU-by-length plot
BUCKETS = [(1, 10), (11, 15), (16, 20), (21, 100)]


def bucket_label(low, high):
    return f"{low}+" if high >= 100 else f"{low}-{high}"


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    src_vocab = Vocab(ckpt["src_itos"])
    tgt_vocab = Vocab(ckpt["tgt_itos"])
    model = Seq2Seq(ckpt["attention"], len(src_vocab), len(tgt_vocab),
                    ckpt["emb_size"], ckpt["hidden_size"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["weights"])
    return model, src_vocab, tgt_vocab


def translate(model, src_sents, tgt_sents, src_vocab, tgt_vocab, device):
    hyps = [None] * len(src_sents)
    for src, lengths, _, idx in make_batches(src_sents, tgt_sents, src_vocab, tgt_vocab,
                                             128, False, device):
        pred, _ = model.decode(src, lengths)
        for row, i in zip(pred.tolist(), idx):
            hyps[i] = " ".join(tgt_vocab.decode(row))
    return hyps


def bleu_score(hyps, refs):
    # text is already lowercased and split into words, so no extra tokenizing
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score


def append_rows(path, header, rows):
    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=["fixed_vector", "additive"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    model, src_vocab, tgt_vocab = load_checkpoint(
        f"checkpoints/{args.attention}_seed{args.seed}.pt", device)
    data, _, _ = load_multi30k("en", "de")
    test_src, test_tgt = data["test"]

    test_loss = average_loss(model, test_src, test_tgt, src_vocab, tgt_vocab, device)
    hyps = translate(model, test_src, test_tgt, src_vocab, tgt_vocab, device)
    refs = [" ".join(t) for t in test_tgt]
    bleu = bleu_score(hyps, refs)

    print(f"{args.attention} seed {args.seed}")
    print(f"test loss {test_loss:.3f} | test ppl {math.exp(test_loss):.1f} | BLEU {bleu:.2f}")

    # BLEU for short, medium and long source sentences
    bucket_rows = []
    print("\nBLEU by source length:")
    for low, high in BUCKETS:
        picked = [i for i, s in enumerate(test_src) if low <= len(s) <= high]
        if not picked:
            continue
        score = bleu_score([hyps[i] for i in picked], [refs[i] for i in picked])
        label = bucket_label(low, high)
        print(f"  {label:>6} words | {len(picked):4d} sentences | BLEU {score:.2f}")
        bucket_rows.append([args.attention, args.seed, label, len(picked), round(score, 2)])

    print("\nsome examples:")
    for i in range(5):
        print("source   :", " ".join(test_src[i]))
        print("reference:", refs[i])
        print("model    :", hyps[i])
        print()

    append_rows("results/test.csv",
                ["attention", "seed", "test_loss", "test_ppl", "bleu"],
                [[args.attention, args.seed, round(test_loss, 4),
                  round(math.exp(test_loss), 2), round(bleu, 2)]])
    append_rows("results/bleu_by_length.csv",
                ["attention", "seed", "bucket", "count", "bleu"], bucket_rows)


if __name__ == "__main__":
    main()