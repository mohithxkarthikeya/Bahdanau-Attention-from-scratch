import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_multi30k, EOS
from src.evaluate import load_checkpoint, BUCKETS, bucket_label
from src.train import get_device


def bleu_vs_length_plot():
    path = "results/bleu_by_length.csv"
    if not os.path.exists(path):
        print("no results/bleu_by_length.csv yet, run src.evaluate first")
        return

    # if the same run was evaluated twice, the later row replaces the earlier one
    latest = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            latest[(row["attention"], row["seed"], row["bucket"])] = float(row["bleu"])

    labels = [bucket_label(low, high) for low, high in BUCKETS]
    plt.figure(figsize=(6, 4))
    for attention in ["fixed_vector", "additive"]:
        means, spreads = [], []
        for label in labels:
            scores = [v for (a, s, b), v in latest.items() if a == attention and b == label]
            means.append(np.mean(scores) if scores else np.nan)
            spreads.append(np.std(scores) if scores else 0)   # spread over seeds
        plt.errorbar(range(len(labels)), means, yerr=spreads, marker="o",
                     capsize=3, label=attention)

    plt.xticks(range(len(labels)), labels)
    plt.xlabel("source sentence length (words)")
    plt.ylabel("BLEU on the test set")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/bleu_vs_length.png", dpi=150)
    plt.close()
    print("saved plots/bleu_vs_length.png")


def draw_heatmap(model, words, src_vocab, tgt_vocab, device, path):
    ids = src_vocab.encode(words)
    src = torch.tensor([ids], device=device)
    lengths = torch.tensor([len(ids)], device=device)
    pred, weights = model.decode(src, lengths)

    # cut the output after the first <eos>
    out_ids = pred[0].tolist()
    if EOS in out_ids:
        out_ids = out_ids[:out_ids.index(EOS) + 1]
    out_words = [tgt_vocab.itos[i] for i in out_ids]
    grid = weights[0, :len(out_ids), :].cpu().numpy()

    fig, ax = plt.subplots(figsize=(0.6 * len(words) + 2, 0.5 * len(out_words) + 2))
    ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90)
    ax.set_yticks(range(len(out_words)))
    ax.set_yticklabels(out_words)
    ax.set_xlabel("source (english)")
    ax.set_ylabel("model output (german)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmaps(seed, count):
    ckpt_path = f"checkpoints/additive_seed{seed}.pt"
    if not os.path.exists(ckpt_path):
        print(f"{ckpt_path} not found, train the additive model first")
        return

    device = get_device()
    model, src_vocab, tgt_vocab = load_checkpoint(ckpt_path, device)
    data, _, _ = load_multi30k("en", "de")
    test_src, _ = data["test"]

    os.makedirs("plots", exist_ok=True)
    for i in range(count):
        path = f"plots/attention_heatmap_{i}.png"
        draw_heatmap(model, test_src[i], src_vocab, tgt_vocab, device, path)
        print("saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    bleu_vs_length_plot()
    heatmaps(args.seed, args.count)


if __name__ == "__main__":
    main()