"""Create exploratory charts from the raw labeled dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "labeled_data.csv"
FIGURES_PATH = ROOT / "reports" / "figures"
LABEL_NAMES = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}


def main() -> None:
    data = pd.read_csv(RAW_PATH)
    data["label"] = data["class"].map(LABEL_NAMES)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")

    counts = data["label"].value_counts().reindex(LABEL_NAMES.values())
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=axis, color=["#d95f02", "#1b9e77", "#7570b3"])
    axis.set_title("Tweet class distribution")
    axis.set_xlabel("")
    axis.set_ylabel("Number of tweets")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    figure.savefig(FIGURES_PATH / "class_distribution.png", dpi=160)
    plt.close(figure)

    lengths = data["tweet"].astype(str).str.split().str.len()
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths, bins=40, ax=axis, color="#2c7fb8")
    axis.set_title("Tweet length distribution")
    axis.set_xlabel("Words per tweet")
    axis.set_ylabel("Tweets")
    figure.tight_layout()
    figure.savefig(FIGURES_PATH / "tweet_length_distribution.png", dpi=160)
    plt.close(figure)

    print(f"Saved visualizations to {FIGURES_PATH}")


if __name__ == "__main__":
    main()
