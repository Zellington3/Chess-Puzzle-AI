"""
scripts/plot_results.py

Generate training curve and ablation comparison bar charts.

Usage:
    python scripts/plot_results.py
"""

import os, sys, json
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_curve(hist_path="outputs/history_full.json", save_path="outputs/training_curve.png"):
    if not os.path.exists(hist_path):
        print(f"No history at {hist_path}")
        return
    with open(hist_path) as f:
        data = json.load(f)
    h = data["history"]
    epochs = [r["epoch"] for r in h]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, [r["train_loss"] for r in h], label="Train Loss")
    ax1.plot(epochs, [r["val_loss"] for r in h], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")

    ax2.plot(epochs, [r["train_acc"] for r in h], label="Train Acc")
    ax2.plot(epochs, [r["val_acc"] for r in h], label="Val Acc")
    ax2.plot(epochs, [r["val_top3"] for r in h], label="Val Top-3", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


def plot_ablation(abl_path="outputs/ablation_results.json", save_path="outputs/ablation_chart.png"):
    if not os.path.exists(abl_path):
        print(f"No ablation results at {abl_path}")
        return
    with open(abl_path) as f:
        results = json.load(f)

    names = [r["model"] for r in results]
    accs = [r["test_acc"] for r in results]
    top3s = [r["test_top3"] for r in results]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - 0.15 for i in x], accs, 0.3, label="Top-1 Accuracy", color="#2563eb")
    ax.bar([i + 0.15 for i in x], top3s, 0.3, label="Top-3 Accuracy", color="#16a34a")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy"); ax.set_title("Ablation Study: Model Comparison")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    plot_training_curve()
    plot_ablation()
