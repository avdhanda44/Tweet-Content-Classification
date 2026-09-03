"""Create a cleaned, model-ready copy without changing the raw dataset."""

from pathlib import Path
import html
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "labeled_data.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "tweets_cleaned.csv"


def clean_tweet(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\brt\b", " RETWEET ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w#'!?\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    data = pd.read_csv(RAW_PATH)
    data = data[["tweet", "class"]].dropna().copy()
    data["class"] = data["class"].astype(int)
    data["clean_tweet"] = data["tweet"].map(clean_tweet)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_PATH, index=False)
    print(f"Read {len(data):,} rows from {RAW_PATH}")
    print(f"Wrote processed data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
