"""
scripts/train.py

Train PuzzleNet on Lichess chess puzzles.

Usage:
    python scripts/train.py                                # defaults (100k puzzles, 20 epochs)
    python scripts/train.py --max_rows 10000 --epochs 5    # quick test
    python scripts/train.py --model small                  # ablation variant
"""

import argparse, os, sys, time, json
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import get_dataloaders
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear, count_params


MODEL_MAP = {
    "full":    PuzzleNet,
    "small":   PuzzleNetSmall,
    "linear":  PuzzleNetLinear,
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for board, meta, target, _ in loader:
        board, meta, target = board.to(device), meta.to(device), target.to(device)

        logits = model(board, meta)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * board.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        total += board.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    top3_correct = 0
    for board, meta, target, _ in loader:
        board, meta, target = board.to(device), meta.to(device), target.to(device)
        logits = model(board, meta)
        loss = criterion(logits, target)

        total_loss += loss.item() * board.size(0)
        correct += (logits.argmax(dim=1) == target).sum().item()
        _, top3 = logits.topk(3, dim=1)
        top3_correct += (top3 == target.unsqueeze(1)).any(dim=1).sum().item()
        total += board.size(0)

    return total_loss / total, correct / total, top3_correct / total


def main():
    parser = argparse.ArgumentParser(description="Train chess puzzle solver")
    parser.add_argument("--data", default="data/puzzles_100k.csv")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--model", choices=["full", "small", "linear"], default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and split data
    train_dl, val_dl, test_dl = get_dataloaders(
        args.data, batch_size=args.batch_size, max_rows=args.max_rows
    )

    # Set up model
    model = MODEL_MAP[args.model]().to(device)
    print(f"Model: {args.model} -- {count_params(model):,} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc, val_top3 = evaluate(model, val_dl, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4), "val_top3": round(val_top3, 4),
            "time_s": round(elapsed, 1),
        }
        history.append(row)
        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_top3={val_top3:.4f}  "
              f"[{elapsed:.1f}s]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model ({val_acc:.4f})")

    # Final test evaluation using best checkpoint
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_acc, test_top3 = evaluate(model, test_dl, criterion, device)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({args.model}):  "
          f"acc={test_acc:.4f}  top3={test_top3:.4f}  loss={test_loss:.4f}")
    print(f"{'='*60}")

    # Save history to JSON
    hist_path = os.path.join(args.save_dir, f"history_{args.model}.json")
    with open(hist_path, "w") as f:
        json.dump({"test_acc": test_acc, "test_top3": test_top3, "history": history}, f, indent=2)
    print(f"Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
