"""
scripts/demo.py

Generate demo output for a 2-5 minute screen recording.
Walks through: data inspection, model info, solving sample puzzles, and metrics.

Usage:
    python scripts/demo.py
"""

import os, sys, json, random, time
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import load_puzzles, encode_board, encode_metadata, encode_move, decode_move
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear, count_params
from scripts.solve_puzzle import print_board, solve_fen


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "data/puzzles_100k.csv"

    # Data Overview
    section("1. DATASET OVERVIEW")
    puzzles = load_puzzles(data_path, max_rows=100000)
    print(f"Total puzzles loaded: {len(puzzles):,}")
    ratings = [p["rating"] for p in puzzles]
    print(f"Rating range: {min(ratings)} to {max(ratings)}")
    print(f"Mean rating: {sum(ratings)/len(ratings):.0f}")
    print(f"\nSample puzzle:")
    p = puzzles[0]
    print(f"  ID: {p['puzzle_id']}")
    print(f"  FEN: {p['fen']}")
    print(f"  Moves: {' '.join(p['moves'])}")
    print(f"  Rating: {p['rating']}")
    print(f"  Themes: {p['themes']}")

    # Model Architecture
    section("2. MODEL ARCHITECTURES (Ablation)")
    for name, cls in [("PuzzleNet (full)", PuzzleNet),
                      ("PuzzleNetSmall", PuzzleNetSmall),
                      ("PuzzleNetLinear", PuzzleNetLinear)]:
        print(f"  {name}: {count_params(cls()):,} parameters")

    # Load trained model and solve puzzles
    section("3. SOLVING SAMPLE PUZZLES")
    ckpt = "outputs/best_full.pt"
    if not os.path.exists(ckpt):
        print(f"  Checkpoint not found at {ckpt}.")
        print(f"  Run training first:  python scripts/train.py")
        return

    model = PuzzleNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    sample = random.sample(puzzles, 5)
    correct = 0
    for i, p in enumerate(sample, 1):
        board = chess.Board(p["fen"])
        board.push(chess.Move.from_uci(p["moves"][0]))
        answer = p["moves"][1]

        pred, top3 = solve_fen(model, board.fen(), device)
        is_correct = pred == answer
        correct += int(is_correct)

        hit = "CORRECT" if is_correct else "WRONG"
        print(f"-- Puzzle {i}: rating {p['rating']} --")
        print_board(board)
        print(f"  Answer:     {answer}")
        print(f"  Prediction: {pred}  [{hit}]")
        print(f"  Top 3: {', '.join(f'{m} ({s:.1%})' for m, s in top3)}\n")

    print(f"Demo accuracy: {correct}/5 ({correct/5:.0%})")

    # Training History
    section("4. TRAINING METRICS")
    hist_path = "outputs/history_full.json"
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            data = json.load(f)
        print(f"  Test accuracy:  {data['test_acc']:.4f}")
        print(f"  Test top-3 acc: {data['test_top3']:.4f}")
        print(f"\n  Epoch  Train Acc  Val Acc  Val Top-3")
        print(f"  {'_'*42}")
        for h in data["history"]:
            print(f"  {h['epoch']:5d}  {h['train_acc']:9.4f}  {h['val_acc']:7.4f}  {h['val_top3']:9.4f}")
    else:
        print(f"  No training history found. Run training first.")

    # Ablation
    section("5. ABLATION RESULTS")
    abl_path = "outputs/ablation_results.json"
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            results = json.load(f)
        print(f"  {'Model':<10} {'Params':>10} {'Test Acc':>10} {'Top-3':>10}")
        print(f"  {'_'*42}")
        for r in results:
            print(f"  {r['model']:<10} {r['params']:>10,} {r['test_acc']:>10.4f} {r['test_top3']:>10.4f}")
    else:
        print("  No ablation results found. Run:  python scripts/ablation.py")

    section("DEMO COMPLETE")


if __name__ == "__main__":
    main()
