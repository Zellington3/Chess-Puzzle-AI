"""
scripts/solve_puzzle.py

Interactive CLI that loads a trained model and solves puzzles.

Usage:
    python scripts/solve_puzzle.py --model full --checkpoint outputs/best_full.pt
    python scripts/solve_puzzle.py --demo --data data/puzzles_100k.csv
"""

import argparse, os, sys, random
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import (
    encode_board, encode_metadata, encode_move, decode_move, load_puzzles
)
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear

MODEL_MAP = {"full": PuzzleNet, "small": PuzzleNetSmall, "linear": PuzzleNetLinear}


def solve_fen(model, fen: str, device) -> tuple:
    """
    Given a FEN (board state after opponent's setup move), predict the best move
    Returns (predicted_move_uci, top_3_moves_with_scores)
    """
    board = chess.Board(fen)
    board_t = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    meta_t = torch.from_numpy(encode_metadata(board)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(board_t, meta_t).squeeze(0)

    # Mask illegal moves so the model only picks from valid options
    legal_indices = [encode_move(m) for m in board.legal_moves]
    mask = torch.full((4096,), float("-inf"), device=device)
    mask[legal_indices] = 0
    logits = logits + mask

    probs = torch.softmax(logits, dim=0)
    top_vals, top_idxs = probs.topk(3)

    top3 = [(decode_move(idx.item()).uci(), val.item()) for idx, val in zip(top_idxs, top_vals)]
    best_move = top3[0][0]
    return best_move, top3


def print_board(board: chess.Board):
    # Print the board with file/rank labels
    print()
    print("    a b c d e f g h")
    print("  +-----------------+")
    for rank in range(7, -1, -1):
        row = f"{rank+1} | "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row += (piece.symbol() if piece else ".") + " "
        row += f"| {rank+1}"
        print(row)
    print("  +-----------------+")
    print("    a b c d e f g h")
    print(f"  {'White' if board.turn == chess.WHITE else 'Black'} to move\n")


def demo_mode(model, data_path, device, n=10):
    # Solve n random puzzles from the dataset and show the results
    puzzles = load_puzzles(data_path)
    sample = random.sample(puzzles, min(n, len(puzzles)))

    correct = 0
    for i, p in enumerate(sample, 1):
        board = chess.Board(p["fen"])
        setup = chess.Move.from_uci(p["moves"][0])
        board.push(setup)
        answer = p["moves"][1]

        pred, top3 = solve_fen(model, board.fen(), device)

        hit = "CORRECT" if pred == answer else "WRONG"
        if pred == answer:
            correct += 1

        print(f"-- Puzzle {i} (rating {p['rating']}, themes: {p['themes']}) --")
        print_board(board)
        print(f"  Correct answer : {answer}")
        print(f"  Model predicted: {pred}  [{hit}]")
        print(f"  Top 3: {', '.join(f'{m} ({s:.1%})' for m, s in top3)}")
        print()

    print(f"Demo result: {correct}/{len(sample)} correct ({correct/len(sample):.0%})")


def interactive_mode(model, device):
    # Let the user paste FEN strings and get predictions
    print("Chess Puzzle Solver -- Interactive Mode")
    print("Enter a FEN string (after opponent's move), or 'quit' to exit.\n")

    while True:
        fen = input("FEN> ").strip()
        if fen.lower() in ("quit", "exit", "q"):
            break
        try:
            board = chess.Board(fen)
            print_board(board)
            pred, top3 = solve_fen(model, fen, device)
            print(f"  Predicted move: {pred}")
            print(f"  Top 3: {', '.join(f'{m} ({s:.1%})' for m, s in top3)}")
            print()
        except Exception as e:
            print(f"  Error: {e}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["full", "small", "linear"], default="full")
    parser.add_argument("--checkpoint", default="outputs/best_full.pt")
    parser.add_argument("--demo", action="store_true", help="Solve random puzzles from dataset")
    parser.add_argument("--data", default="data/puzzles_100k.csv")
    parser.add_argument("--n", type=int, default=10, help="Number of demo puzzles")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_MAP[args.model]().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {args.model} model from {args.checkpoint}\n")

    if args.demo:
        demo_mode(model, args.data, device, n=args.n)
    else:
        interactive_mode(model, device)


if __name__ == "__main__":
    main()
