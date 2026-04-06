"""
scripts/evaluate.py

Evaluate a trained model and print detailed metrics including
accuracy broken down by puzzle rating.

Usage:
    python scripts/evaluate.py --model full --checkpoint outputs/best_full.pt
"""

import argparse, os, sys
import torch
import torch.nn as nn
import chess
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import (
    get_dataloaders, load_puzzles, decode_move, encode_board,
    encode_metadata, encode_move
)
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear

MODEL_MAP = {"full": PuzzleNet, "small": PuzzleNetSmall, "linear": PuzzleNetLinear}


def legal_move_accuracy(model, puzzles, device, max_n=2000):
    """
    Evaluate accuracy while restricting predictions to legal moves only
    This is a fairer metric since the raw model output can include illegal moves
    """
    model.eval()
    correct, total = 0, 0
    for p in puzzles[:max_n]:
        try:
            board = chess.Board(p["fen"])
            board.push(chess.Move.from_uci(p["moves"][0]))
            answer = chess.Move.from_uci(p["moves"][1])

            board_t = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
            meta_t = torch.from_numpy(encode_metadata(board)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(board_t, meta_t).squeeze(0)

            # Set illegal move logits to -inf so they can't be selected
            legal_indices = [encode_move(m) for m in board.legal_moves]
            mask = torch.full((4096,), float("-inf"), device=device)
            mask[legal_indices] = 0
            logits = logits + mask

            pred_idx = logits.argmax().item()
            pred_move = decode_move(pred_idx)

            if pred_move == answer:
                correct += 1
            total += 1
        except Exception:
            continue

    return correct / total if total > 0 else 0.0


def accuracy_by_rating(model, puzzles, device, max_n=5000):
    # Break down accuracy into puzzle rating buckets (500-wide)
    model.eval()
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for p in puzzles[:max_n]:
        try:
            board = chess.Board(p["fen"])
            board.push(chess.Move.from_uci(p["moves"][0]))
            answer = chess.Move.from_uci(p["moves"][1])

            board_t = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
            meta_t = torch.from_numpy(encode_metadata(board)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(board_t, meta_t).squeeze(0)

            legal_indices = [encode_move(m) for m in board.legal_moves]
            mask = torch.full((4096,), float("-inf"), device=device)
            mask[legal_indices] = 0
            logits = logits + mask

            pred_idx = logits.argmax().item()
            pred_move = decode_move(pred_idx)

            rating = p["rating"]
            bucket = f"{(rating // 500) * 500}-{(rating // 500) * 500 + 499}"
            buckets[bucket]["total"] += 1
            if pred_move == answer:
                buckets[bucket]["correct"] += 1
        except Exception:
            continue

    return {k: v["correct"] / v["total"] for k, v in sorted(buckets.items()) if v["total"] > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["full", "small", "linear"], default="full")
    parser.add_argument("--checkpoint", default="outputs/best_full.pt")
    parser.add_argument("--data", default="data/puzzles_100k.csv")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_MAP[args.model]().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    puzzles = load_puzzles(args.data, max_rows=args.max_rows)

    # Overall legal-move accuracy
    acc = legal_move_accuracy(model, puzzles, device)
    print(f"Legal-move accuracy: {acc:.4f}")

    # Accuracy by rating bucket
    print("\nAccuracy by puzzle rating:")
    for bucket, acc_val in accuracy_by_rating(model, puzzles, device).items():
        print(f"  {bucket}: {acc_val:.4f}")


if __name__ == "__main__":
    main()
