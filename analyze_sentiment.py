#!/usr/bin/env python3
import json, os, argparse
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from tqdm import tqdm
import numpy as np

download('vader_lexicon', quiet=True)

def analyze_sentiment(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sia = SentimentIntensityAnalyzer()

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing reviews"):
            rv = json.loads(line)
            text = (rv.get("review") or "").strip()
            if not text:
                continue
            s = sia.polarity_scores(text)
            records.append({
                "review_id": rv.get("recommendationid"),
                "review": text,
                "voted_up": rv.get("voted_up"),
                "sentiment": s["compound"],
                "pos": s["pos"],
                "neu": s["neu"],
                "neg": s["neg"],
                "votes_up": rv.get("votes_up"),
                "votes_funny": rv.get("votes_funny"),
                "playtime_forever": rv.get("author", {}).get("playtime_forever", 0)
            })



    df = pd.DataFrame(records)
    out_csv = os.path.join(output_dir, "sentiment_results.csv")
    df.to_csv(out_csv, index=False)
    # Compute numeric summary
    stats = df.describe(include=[float, int])

    # Interpret the data
    mean_sent = stats.loc["mean", "sentiment"]
    pos_share = (df["sentiment"] > 0.2).mean()
    neg_share = (df["sentiment"] < -0.2).mean()

    summary_lines = [
        f"Total analyzed reviews: {len(df):,}",
        f"Average sentiment: {mean_sent:.3f}",
        f"Positive share (>0.2): {pos_share * 100:.1f}%",
        f"Negative share (<-0.2): {neg_share * 100:.1f}%",
        "",
        f"Mean positive word ratio: {stats.loc['mean', 'pos']:.3f}",
        f"Mean neutral ratio: {stats.loc['mean', 'neu']:.3f}",
        f"Mean negative ratio: {stats.loc['mean', 'neg']:.3f}",
        "",
        f"Average upvotes per review: {stats.loc['mean', 'votes_up']:.2f}",
        f"Average funny votes per review: {stats.loc['mean', 'votes_funny']:.2f}",
        f"Median playtime before review (hrs): {df['playtime_forever'].median() / 60:.1f}",
        "",
        f"Sentiment std deviation: {stats.loc['std', 'sentiment']:.3f} (spread of opinions)",
        f"Most negative review sentiment: {stats.loc['min', 'sentiment']:.3f}",
        f"Most positive review sentiment: {stats.loc['max', 'sentiment']:.3f}",
    ]

    out_summary = os.path.join(output_dir, "summary.txt")
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Summary written to {out_summary}")
    print(f"Saved {len(df)} analyzed reviews → {out_csv}")
    print(df.describe(include=[np.number]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out_reviews/reviews_1808500.jsonl", help="Path to reviews_<appid>.jsonl")
    ap.add_argument("--outdir", default="analysis_out")
    args = ap.parse_args()
    analyze_sentiment(args.input, args.outdir)
