#!/usr/bin/env python3
"""
get_insights.py
---------------
Extracts key topics from sentiment_results.csv using TF-IDF+KMeans or BERTopic.

Input:
    analysis_out/sentiment_results.csv
Output:
    analysis_out/review_insights.txt
"""
import os, argparse, re
import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk

# ---------- Optional BERTopic ------------
USE_BERTOPIC = False   # set True if you have the package installed
if USE_BERTOPIC:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
# ----------------------------------------

DetectorFactory.seed = 0
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV must include 'review' and 'sentiment' columns.")
    df["is_english"] = [
        is_english(t) if isinstance(t, str) and len(t) > 20 else False
        for t in tqdm(df["review"], desc="Language detection")
    ]
    df = df[df["is_english"]]
    print(f"Remaining English reviews: {len(df)}")
    return df

# ---------- TF-IDF + KMeans ------------
def tfidf_kmeans_topics(texts, n_clusters=8, top_n=10):
    vec = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    terms = vec.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]
    topics = []
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order[i, :top_n]]
        topics.append(", ".join(top_terms))
    return topics
# ---------------------------------------

def bertopic_topics(texts, min_topic_size=20, top_n=10):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=False)
    topics, _ = topic_model.fit_transform(texts, model.encode(texts, show_progress_bar=True))
    unique_topics = sorted(set(t for t in topics if t != -1))
    topic_map = []
    for t in unique_topics:
        words = [w.split(':')[0] for w in topic_model.get_topic(t)[:top_n]]
        topic_map.append(", ".join(words))
    return topic_map

def main():
    ap = argparse.ArgumentParser(description="Generate insights from sentiment data")
    ap.add_argument("--input", default="analysis_out/sentiment_results.csv")
    ap.add_argument("--outdir", default="analysis_out")
    ap.add_argument("--clusters", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.input)

    total = len(df)
    mean_sent = df["sentiment"].mean()
    pos_share = (df["sentiment"] > 0.2).mean() * 100
    neg_share = (df["sentiment"] < -0.2).mean() * 100

    pos_texts = df.loc[df["sentiment"] > 0.2, "review"].astype(str).tolist()
    neg_texts = df.loc[df["sentiment"] < -0.2, "review"].astype(str).tolist()

    print(f"\nExtracting topics using {'BERTopic' if USE_BERTOPIC else 'TF-IDF + KMeans'} ...")

    if USE_BERTOPIC:
        pos_topics = bertopic_topics(pos_texts, top_n=10)
        neg_topics = bertopic_topics(neg_texts, top_n=10)
    else:
        pos_topics = tfidf_kmeans_topics(pos_texts, n_clusters=args.clusters)
        neg_topics = tfidf_kmeans_topics(neg_texts, n_clusters=args.clusters)

    report_path = os.path.join(args.outdir, "review_insights.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Arc Raiders – Review Insights\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total reviews analyzed: {total:,}\n")
        f.write(f"Average sentiment: {mean_sent:.3f}\n")
        f.write(f"Positive share (>0.2): {pos_share:.1f}%\n")
        f.write(f"Negative share (<-0.2): {neg_share:.1f}%\n\n")

        f.write("Top Positive Topics\n-------------------\n")
        for t in pos_topics:
            f.write(f"- {t}\n")

        f.write("\nTop Negative Topics\n-------------------\n")
        for t in neg_topics:
            f.write(f"- {t}\n")

        f.write("\nRecommended Developer Focus\n---------------------------\n")
        focus = "performance optimization / stability" if any(
            kw in " ".join(neg_topics).lower()
            for kw in ["performance", "fps", "lag", "stutter", "crash", "server"]
        ) else "gameplay balance and content depth" if any(
            kw in " ".join(neg_topics).lower()
            for kw in ["balance", "mission", "progression", "content"]
        ) else "general polish and UX"
        f.write(f"Next sprint should prioritize **{focus}** based on most frequent negative topic terms.\n")

    print(f"\nInsight report written → {report_path}")

if __name__ == "__main__":
    main()
