"""
tests/test_puzzles.py

Unit tests for data preprocessing and model components.

Usage:
    python -m pytest tests/test_puzzles.py -v
    python tests/test_puzzles.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import chess
import numpy as np
from data.preprocess import (
    encode_board, encode_metadata, encode_move, decode_move,
    load_puzzles, puzzles_to_records
)
from models.puzzle_net import PuzzleNet, PuzzleNetSmall, PuzzleNetLinear


def test_encode_board_shape():
    board = chess.Board()
    t = encode_board(board)
    assert t.shape == (12, 8, 8), f"Expected (12,8,8), got {t.shape}"
    # Starting position has 32 pieces on the board
    assert t.sum() == 32, f"Expected 32 pieces, got {t.sum()}"
    print("[PASS] encode_board shape and piece count correct")


def test_encode_board_empty():
    board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
    t = encode_board(board)
    assert t.sum() == 0, "Empty board should have all zeros"
    print("[PASS] encode_board empty board correct")


def test_encode_decode_move():
    move = chess.Move.from_uci("e2e4")
    idx = encode_move(move)
    decoded = decode_move(idx)
    assert decoded.from_square == move.from_square
    assert decoded.to_square == move.to_square
    print(f"[PASS] encode/decode move: e2e4 -> {idx} -> {decoded.uci()}")


def test_metadata_shape():
    board = chess.Board()
    meta = encode_metadata(board)
    assert meta.shape == (5,), f"Expected (5,), got {meta.shape}"
    assert meta[0] == 1.0, "White to move should be 1.0"
    print("[PASS] metadata shape and values correct")


def test_model_forward_pass():
    for name, cls in [("PuzzleNet", PuzzleNet), ("Small", PuzzleNetSmall), ("Linear", PuzzleNetLinear)]:
        model = cls()
        board = torch.randn(2, 12, 8, 8)
        meta = torch.randn(2, 5)
        out = model(board, meta)
        assert out.shape == (2, 4096), f"{name}: Expected (2,4096), got {out.shape}"
        print(f"[PASS] {name} forward pass -> {out.shape}")


def test_load_puzzles():
    path = "data/puzzles_100k.csv"
    if not os.path.exists(path):
        print("[SKIP] test_load_puzzles (data file not found)")
        return
    puzzles = load_puzzles(path, max_rows=10)
    assert len(puzzles) == 10
    assert "fen" in puzzles[0]
    assert "moves" in puzzles[0]
    assert isinstance(puzzles[0]["moves"], list)
    print(f"[PASS] Loaded 10 puzzles, first has {len(puzzles[0]['moves'])} moves")


def test_puzzles_to_records():
    path = "data/puzzles_100k.csv"
    if not os.path.exists(path):
        print("[SKIP] test_puzzles_to_records (data file not found)")
        return
    puzzles = load_puzzles(path, max_rows=50)
    records = puzzles_to_records(puzzles)
    assert len(records) > 0
    board_np, meta_np, move_idx, rating = records[0]
    assert board_np.shape == (12, 8, 8)
    assert meta_np.shape == (5,)
    assert 0 <= move_idx < 4096
    print(f"[PASS] Converted {len(records)} puzzles to records")


def test_legal_move_prediction():
    # Make sure we can encode and decode all legal moves for a position
    board = chess.Board()
    legal = list(board.legal_moves)
    indices = [encode_move(m) for m in legal]
    # All indices should be in range
    for idx in indices:
        assert 0 <= idx < 4096
    # Decoded moves should have valid squares
    for idx in indices:
        m = decode_move(idx)
        assert 0 <= m.from_square < 64
        assert 0 <= m.to_square < 64
    print(f"[PASS] Legal move masking: {len(legal)} moves encoded/decoded")


if __name__ == "__main__":
    test_encode_board_shape()
    test_encode_board_empty()
    test_encode_decode_move()
    test_metadata_shape()
    test_model_forward_pass()
    test_load_puzzles()
    test_puzzles_to_records()
    test_legal_move_prediction()
    print("\nAll tests passed!")
