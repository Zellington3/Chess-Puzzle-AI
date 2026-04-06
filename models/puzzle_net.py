"""
models/puzzle_net.py

CNN-based model that predicts the best move from a board position.

Architecture overview:
  - Input: 12-channel 8x8 board + 5-dim metadata vector
  - Several residual conv blocks pull out spatial features
  - Global average pool, concat with metadata, FC head, 4096-class output

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_MOVE_CLASSES = 4096


class ResBlock(nn.Module):
    # Standard residual block: conv, BN, ReLU, conv, BN, skip add, ReLU.

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PuzzleNet(nn.Module):
    """
    The main model. Default config: 128 filters, 6 residual blocks, 512-wide FC head.
    """

    def __init__(self, num_filters=128, num_blocks=6, fc_size=512):
        super().__init__()

        # First convolution: 12 input channels to num_filters
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.res_tower = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head: reduce channels before flattening
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Fully connected layers: 32*8*8 + 5 metadata features -> fc_size -> 4096
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 5, fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_size, NUM_MOVE_CLASSES),
        )

    def forward(self, board, meta):
        """
        board: (B, 12, 8, 8)
        meta:  (B, 5)
        Returns: (B, 4096) logits over possible moves
        """
        x = self.input_conv(board)
        x = self.res_tower(x)
        x = self.policy_conv(x)
        x = x.view(x.size(0), -1)          # (B, 32*64)
        x = torch.cat([x, meta], dim=1)    # (B, 32*64 + 5)
        return self.fc(x)


class PuzzleNetSmall(nn.Module):
    # Smaller variant for ablation: fewer filters and blocks

    def __init__(self, num_filters=64, num_blocks=3, fc_size=256):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_blocks)]
        )
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 5, fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_size, NUM_MOVE_CLASSES),
        )

    def forward(self, board, meta):
        x = self.input_conv(board)
        x = self.res_tower(x)
        x = self.policy_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, meta], dim=1)
        return self.fc(x)


class PuzzleNetLinear(nn.Module):
    # Baseline with no convolutions at all. Just flatten and classify

    def __init__(self, fc_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(12 * 8 * 8 + 5, fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_size, NUM_MOVE_CLASSES),
        )

    def forward(self, board, meta):
        x = board.view(board.size(0), -1)
        x = torch.cat([x, meta], dim=1)
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, cls in [("PuzzleNet", PuzzleNet), ("PuzzleNetSmall", PuzzleNetSmall), ("PuzzleNetLinear", PuzzleNetLinear)]:
        m = cls()
        print(f"{name}: {count_params(m):,} trainable parameters")
