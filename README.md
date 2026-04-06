# Chess Puzzle-Solving AI

A complete, runnable AI system that solves chess puzzles. Trained from scratch
on the Lichess Puzzle Database.

Dataset download: https://database.lichess.org/lichess_db_puzzle.csv.zst

## What This Project Does

Given a chess position, the model predicts the correct first move of a tactical
puzzle. The project includes:

- A neural network trained from scratch (CNN with residual blocks)
- An ablation study comparing 3 architectures (Linear, Small CNN, Full CNN)
- A Stockfish baseline for comparison
- An interactive CLI to solve puzzles
- An evaluation pipeline with accuracy-by-rating breakdowns

## Project Structure

```
chess-puzzle-ai/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- preprocess.py          # Data loading, board encoding, train/val/test split
|   |-- puzzles_100k.csv       # Dataset (see setup instructions below)
|-- models/
|   |-- puzzle_net.py           # PuzzleNet, PuzzleNetSmall, PuzzleNetLinear
|-- scripts/
|   |-- train.py                # Training pipeline
|   |-- evaluate.py             # Detailed evaluation and accuracy-by-rating
|   |-- ablation.py             # Ablation study (3 model variants)
|   |-- solve_puzzle.py         # Interactive CLI puzzle solver
|   |-- stockfish_baseline.py   # Stockfish comparison
|   |-- demo.py                 # Full demo for screen recording
|   |-- plot_results.py         # Generate training curve and ablation charts
|-- tests/
|   |-- test_puzzles.py         # Unit tests
|-- outputs/                    # Saved models, metrics, plots (created automatically)
```

## Setup

### 1. Get the dataset

Download the Lichess puzzle database from:
https://database.lichess.org/lichess_db_puzzle.csv.zst

Decompress it and take the first 100k rows:

Windows (PowerShell):
```powershell
.\zstd.exe -d lichess_db_puzzle.csv.zst -o lichess_db_puzzle.csv
Get-Content lichess_db_puzzle.csv -TotalCount 100001 | Set-Content data\puzzles_100k.csv
```

Mac/Linux:
```bash
zstd -d lichess_db_puzzle.csv.zst
head -n 100001 lichess_db_puzzle.csv > data/puzzles_100k.csv
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU and want faster training:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 3. Verify the setup

```bash
python tests/test_puzzles.py
```

## Running the Project

### Train the model

```bash
# Full training (100k puzzles, about 4 min on GPU, about 30-60 min on CPU)
python scripts/train.py

# Quick test run
python scripts/train.py --max_rows 10000 --epochs 5
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/best_full.pt
```

### Solve puzzles interactively

```bash
# Demo mode: solve 10 random puzzles
python scripts/solve_puzzle.py --demo

# Interactive mode: paste FEN strings
python scripts/solve_puzzle.py
```

### Run ablation study

```bash
python scripts/ablation.py --epochs 15
```

### Generate plots

```bash
python scripts/plot_results.py
```

### Full demo (for video recording)

```bash
python scripts/demo.py
```

### Stockfish baseline (optional)

First install Stockfish from https://stockfishchess.org/download/

Then run with the path to the binary:

Windows:
```bash
python scripts/stockfish_baseline.py --stockfish_path "C:\Users\YourName\stockfish\stockfish-windows-x86-64-avx2.exe" --n 200
```

Mac (Homebrew):
```bash
brew install stockfish
python scripts/stockfish_baseline.py --stockfish_path /opt/homebrew/bin/stockfish --n 200
```

Linux:
```bash
sudo apt install stockfish
python scripts/stockfish_baseline.py --stockfish_path /usr/games/stockfish --n 200
```

## How It Works

### Data Encoding

Each puzzle is encoded as:
- Board tensor: 12-channel 8x8 array (one channel per piece type per color)
- Metadata: 5-dim vector (whose turn it is + 4 castling rights)
- Target: the correct first response move, encoded as from_square * 64 + to_square (4096 classes)

The opponent's setup move is applied first, then the model predicts the player's answer.
The data is automatically split 80/10/10 into train, validation, and test sets.

### Model Architecture (PuzzleNet)

```
Input (12x8x8 board)
  -> Conv2d(12 -> 128) + BN + ReLU
  -> 6x Residual Blocks (conv -> BN -> ReLU -> conv -> BN -> skip)
  -> Policy Conv (128 -> 32) + BN + ReLU
  -> Flatten -> concat with 5-dim metadata
  -> FC(2053 -> 512) -> ReLU -> Dropout
  -> FC(512 -> 4096)  <- move prediction logits
```

At inference time, illegal moves are masked to -inf before picking the top prediction.

### Ablation Variants

| Model | Description |
|-------|-------------|
| PuzzleNetLinear | No convolutions, pure MLP baseline |
| PuzzleNetSmall | 3 res-blocks, 64 filters |
| PuzzleNet | 6 res-blocks, 128 filters (default) |

### Metrics

- Top-1 Accuracy: does the model's best prediction match the puzzle answer?
- Top-3 Accuracy: is the answer in the model's top 3 predictions?
- Legal-move Accuracy: same as Top-1 but after masking out illegal moves
- Accuracy by Rating: breakdown across difficulty buckets (0-499, 500-999, etc.)

## Results

Trained on 100k puzzles for 20 epochs on an NVIDIA RTX 2080 (~13s per epoch).

### Test Set Performance (Full Model)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 37.8% |
| Top-3 Accuracy | 53.9% |
| Legal-Move Accuracy | 83.7% |

### Ablation Study

| Model | Params | Top-1 Acc | Top-3 Acc | Train Time |
|-------|--------|-----------|-----------|------------|
| Linear (MLP) | 2,760,192 | 30.3% | 46.2% | 40s |
| Small CNN | 1,810,080 | 32.5% | 48.6% | 72s |
| Full CNN | 4,945,376 | 35.4% | 52.1% | 198s |

### Accuracy by Puzzle Rating

| Rating | Accuracy |
|--------|----------|
| 0-499 | 97.9% |
| 500-999 | 91.7% |
| 1000-1499 | 85.6% |
| 1500-1999 | 79.1% |
| 2000-2499 | 73.2% |
| 2500+ | 74.9% |

### Live Demo

8/10 puzzles solved correctly in random sample testing.

## Testing

```bash
python tests/test_puzzles.py
```

What gets tested:
- Board encoding shape and piece counts
- Move encoding/decoding roundtrip
- Metadata shape and values
- Forward pass for all 3 model architectures
- Data loading from CSV
- Legal move masking

## Improvements (given more time)

1. Multi-move prediction: predict the full solution sequence, not just the first move
2. Transformer architecture for better pattern recognition
3. Train on the full 4M puzzle dataset
4. Curriculum learning: train on easy puzzles first, then harder ones
5. Hybrid approach: use the neural net to narrow candidates, then verify with Stockfish
6. Data augmentation: board flipping for color-symmetric positions
7. Richer move encoding that handles promotions and distinguishes piece types
8. A Visual Learning model that can record the users screen and determine where the pieces are on the board based on what it sees
