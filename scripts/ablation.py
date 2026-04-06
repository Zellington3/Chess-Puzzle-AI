"""
scripts/ablation.py

Trains all three model variants side by side and saves a comparison summary.

Usage:
    python scripts/ablation.py                              # full 100k dataset
    python scripts/ablation.py --max_rows 20000 --epochs 10 # quick run
"""

import argparse, os, sys, json, time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import get_dataloaders
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear, count_params
from scripts.train import train_one_epoch, evaluate

VARIANTS = [
    ("linear",  PuzzleNetLinear, "No convolutions, pure MLP baseline"),
    ("small",   PuzzleNetSmall,  "3 res-blocks, 64 filters"),
    ("full",    PuzzleNet,       "6 res-blocks, 128 filters (default)"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/puzzles_100k.csv")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data once and reuse for all three variants
    train_dl, val_dl, test_dl = get_dataloaders(
        args.data, batch_size=args.batch_size, max_rows=args.max_rows
    )

    results = []
    for name, model_cls, desc in VARIANTS:
        print(f"\n{'='*60}")
        print(f"ABLATION: {name} -- {desc}")
        print(f"{'='*60}")

        model = model_cls().to(device)
        params = count_params(model)
        print(f"Parameters: {params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0.0
        t_start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_loss, val_acc, val_top3 = evaluate(model, val_dl, criterion, device)

            print(f"  Epoch {epoch:2d}  train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  val_top3={val_top3:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f"best_{name}.pt"))

        # Load best checkpoint and evaluate on test set
        model.load_state_dict(torch.load(
            os.path.join(args.save_dir, f"best_{name}.pt"), weights_only=True
        ))
        test_loss, test_acc, test_top3 = evaluate(model, test_dl, criterion, device)
        elapsed = time.time() - t_start

        results.append({
            "model": name,
            "description": desc,
            "params": params,
            "test_acc": round(test_acc, 4),
            "test_top3": round(test_top3, 4),
            "best_val_acc": round(best_val_acc, 4),
            "train_time_s": round(elapsed, 1),
        })

    # Print summary table
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':>10} {'Test Acc':>10} {'Test Top-3':>12} {'Time (s)':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<10} {r['params']:>10,} {r['test_acc']:>10.4f} "
              f"{r['test_top3']:>12.4f} {r['train_time_s']:>10.1f}")

    # Save to JSON
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
