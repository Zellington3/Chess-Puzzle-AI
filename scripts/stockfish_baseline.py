"""
scripts/stockfish_baseline.py

Compare the neural net against Stockfish on puzzle solving.

You need Stockfish installed. Download it from https://stockfishchess.org/download/

Then pass the path to the binary:

  Windows:
    python scripts/stockfish_baseline.py --stockfish_path "C:\Users\YourName\stockfish\stockfish-windows-x86-64-avx2.exe"

  Mac (Homebrew):
    brew install stockfish
    python scripts/stockfish_baseline.py --stockfish_path /opt/homebrew/bin/stockfish

  Linux:
    sudo apt install stockfish
    python scripts/stockfish_baseline.py --stockfish_path /usr/games/stockfish

Usage:
    python scripts/stockfish_baseline.py --n 200
"""

import argparse, os, sys, time, random
import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import load_puzzles


def solve_with_stockfish(engine, board: chess.Board, time_limit=0.1):
    # Get Stockfish's best move for this position
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/puzzles_100k.csv")
    parser.add_argument("--n", type=int, default=200, help="Number of puzzles to test")
    parser.add_argument("--stockfish_path", default="stockfish",
                        help="Full path to the Stockfish binary (see examples in docstring)")
    parser.add_argument("--time_limit", type=float, default=0.1,
                        help="Seconds per puzzle for Stockfish")
    args = parser.parse_args()

    puzzles = load_puzzles(args.data)
    sample = random.sample(puzzles, min(args.n, len(puzzles)))

    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except Exception as e:
        print(f"Could not start Stockfish at: {args.stockfish_path}")
        print(f"Error: {e}")
        print()
        print("Make sure Stockfish is installed and pass the correct path:")
        print('  Windows: --stockfish_path "C:\\Users\\YourName\\stockfish\\stockfish-windows-x86-64-avx2.exe"')
        print("  Mac:     --stockfish_path /opt/homebrew/bin/stockfish")
        print("  Linux:   --stockfish_path /usr/games/stockfish")
        return

    correct = 0
    total = 0
    t_start = time.time()

    for p in sample:
        try:
            board = chess.Board(p["fen"])
            board.push(chess.Move.from_uci(p["moves"][0]))
            answer = chess.Move.from_uci(p["moves"][1])

            pred = solve_with_stockfish(engine, board, args.time_limit)
            if pred == answer:
                correct += 1
            total += 1
        except Exception:
            continue

    elapsed = time.time() - t_start
    engine.quit()

    print(f"\nStockfish Baseline Results (time_limit={args.time_limit}s)")
    print(f"  Puzzles tested: {total}")
    print(f"  Correct: {correct}/{total} ({correct/total:.1%})")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/total:.3f}s per puzzle)")
    print(f"\nNote: Stockfish sometimes finds an equally good alternative move")
    print(f"that does not match the puzzle's listed answer.")


if __name__ == "__main__":
    main()
