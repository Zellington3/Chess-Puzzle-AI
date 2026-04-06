"""
data/preprocess.py

Loads the Lichess puzzle CSV, encodes board positions as tensors,
and splits the data into train/val/test sets.

Each puzzle row has:
  - FEN: the board position before the puzzle starts
  - Moves: space-separated UCI moves. The FIRST move is the opponent's
    setup move. The SECOND move is the correct answer we want to predict.

Board encoding: 12-channel 8x8 tensor (one channel per piece type per color).
Move encoding: index into a 4096-class space (from_square * 64 + to_square).
"""

import os, csv, random
import numpy as np
import chess
import torch
from torch.utils.data import Dataset, DataLoader

# Constants

PIECE_CHANNELS = {
    (chess.PAWN,   chess.WHITE): 0,  (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.WHITE): 1,  (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.WHITE): 2,  (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.WHITE): 3,  (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.WHITE): 4,  (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.WHITE): 5,  (chess.KING,   chess.BLACK): 11,
}

NUM_MOVE_CLASSES = 4096  # 64 * 64 (from_sq * 64 + to_sq)


# Encoding helpers

def encode_board(board: chess.Board) -> np.ndarray:
    "Turns a chess.Board into a (12, 8, 8) float32 array."
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            ch = PIECE_CHANNELS[(piece.piece_type, piece.color)]
            row, col = divmod(sq, 8)
            tensor[ch, row, col] = 1.0
    return tensor


def encode_move(move: chess.Move) -> int:
    # Convert a UCI move to an integer in [0, 4095]
    return move.from_square * 64 + move.to_square


def decode_move(idx: int) -> chess.Move:
    # Convert an integer back to a chess.Move (no promotion info)
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


def encode_metadata(board: chess.Board) -> np.ndarray:
    "Extra features: whose turn it is (1=white, 0=black) and castling rights (4 bits)."
    meta = np.zeros(5, dtype=np.float32)
    meta[0] = 1.0 if board.turn == chess.WHITE else 0.0
    meta[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    meta[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    meta[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    meta[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return meta


# Dataset

class PuzzleDataset(Dataset):
    """
    Each sample is (board_tensor, metadata, target_move_idx, rating)
    The board state is AFTER the opponent's setup move has been played
    The target is the player's correct first response
    """

    def __init__(self, records: list):
        self.records = records  # list of (board_np, meta_np, move_idx, rating)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        board_np, meta_np, move_idx, rating = self.records[idx]
        return (
            torch.from_numpy(board_np),
            torch.from_numpy(meta_np),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float32),
        )


# Loading and splitting

def load_puzzles(csv_path: str, max_rows: int = None):
    """
    Parse the Lichess CSV and return a list of dicts with keys:
      puzzle_id, fen, moves (list of strings), rating, themes
    """
    puzzles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            puzzles.append({
                "puzzle_id": row["PuzzleId"],
                "fen": row["FEN"],
                "moves": row["Moves"].split(),
                "rating": int(row["Rating"]),
                "themes": row["Themes"],
            })
    return puzzles


def puzzles_to_records(puzzles: list):
    """
    Convert raw puzzle dicts into encoded (board, meta, move_idx, rating) tuples
    Skips any puzzle that fails to parse
    """
    records = []
    skipped = 0
    for p in puzzles:
        try:
            board = chess.Board(p["fen"])
            # Play the opponent's setup move first
            setup_move = chess.Move.from_uci(p["moves"][0])
            board.push(setup_move)
            # The answer is the next move in the list
            answer_move = chess.Move.from_uci(p["moves"][1])
            board_np = encode_board(board)
            meta_np = encode_metadata(board)
            move_idx = encode_move(answer_move)
            records.append((board_np, meta_np, move_idx, p["rating"]))
        except Exception:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} puzzles due to parse errors.")
    return records


def split_data(records, train_frac=0.8, val_frac=0.1, seed=42):
    # Shuffle and split into train / val / test sets
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return records[:t1], records[t1:t2], records[t2:]


def get_dataloaders(csv_path: str, batch_size=256, max_rows=None):
    #Full pipeline: CSV to DataLoaders.
    print(f"Loading puzzles from {csv_path} ...")
    puzzles = load_puzzles(csv_path, max_rows=max_rows)
    print(f"  Loaded {len(puzzles)} puzzles. Encoding boards ...")
    records = puzzles_to_records(puzzles)
    print(f"  Encoded {len(records)} records. Splitting ...")
    train_r, val_r, test_r = split_data(records)
    print(f"  Train: {len(train_r)}  Val: {len(val_r)}  Test: {len(test_r)}")

    train_dl = DataLoader(PuzzleDataset(train_r), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(PuzzleDataset(val_r),   batch_size=batch_size)
    test_dl  = DataLoader(PuzzleDataset(test_r),  batch_size=batch_size)
    return train_dl, val_dl, test_dl


# Quick test from the command line
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/puzzles_100k.csv"
    train_dl, val_dl, test_dl = get_dataloaders(path, max_rows=1000)
    board, meta, target, rating = next(iter(train_dl))
    print(f"Batch shapes: board={board.shape}, meta={meta.shape}, "
          f"target={target.shape}, rating={rating.shape}")
