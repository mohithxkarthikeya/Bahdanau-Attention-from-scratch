import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from src.data import load_multi30k, make_batches, PAD
from src.model import Seq2Seq

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def average_loss(model, src_sents, tgt_sents, src_vocab, tgt_vocab, device):
    # loss per word over a whole split (used for validation and test)
    model.eval()
    total_loss, total_words = 0.0, 0
    for src, lengths, tgt, _ in make_batches(src_sents, tgt_sents, src_vocab, tgt_vocab,
                                             128, False, device):
        logits = model(src, lengths, tgt)
        labels = tgt[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=PAD, reduction="sum")
        total_loss += loss.item()
        total_words += (labels != PAD).sum().item()
    return total_loss / total_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=["fixed_vector", "additive"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data, src_vocab, tgt_vocab = load_multi30k("en", "de")
    train_src, train_tgt = data["train"]
    val_src, val_tgt = data["validation"]
    print(f"device: {device}")
    print(f"train {len(train_src)}, val {len(val_src)}, test {len(data['test'][0])}")
    print(f"vocab: english {len(src_vocab)}, german {len(tgt_vocab)}")

    model = Seq2Seq(args.attention, len(src_vocab), len(tgt_vocab),
                    args.emb_size, args.hidden_size, args.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{args.attention}_seed{args.seed}.pt"
    best_val, bad_epochs, epochs_run = float("inf"), 0, 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total, count = 0.0, 0
        for src, lengths, tgt, _ in make_batches(train_src, train_tgt, src_vocab, tgt_vocab,
                                                 args.batch_size, True, device):
            logits = model(src, lengths, tgt)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt[:, 1:].reshape(-1), ignore_index=PAD)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            count += 1

        val_loss = average_loss(model, val_src, val_tgt, src_vocab, tgt_vocab, device)
        epochs_run = epoch
        print(f"epoch {epoch:2d} | train {total / count:.3f} | val {val_loss:.3f} "
              f"| val ppl {math.exp(val_loss):.1f} | {time.time() - start:.0f}s")

        # keep the best model by validation loss, stop if it does not improve
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({
                "attention": args.attention,
                "emb_size": args.emb_size,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
                "src_itos": src_vocab.itos,
                "tgt_itos": tgt_vocab.itos,
                "weights": model.state_dict(),
            }, path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("validation loss stopped improving, stopping early")
                break

    print(f"best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f}), saved to {path}")

    os.makedirs("results", exist_ok=True)
    new_file = not os.path.exists("results/runs.csv")
    with open("results/runs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["attention", "seed", "epochs_run", "best_val_loss", "best_val_ppl"])
        writer.writerow([args.attention, args.seed, epochs_run,
                         round(best_val, 4), round(math.exp(best_val), 2)])


if __name__ == "__main__":
    main()