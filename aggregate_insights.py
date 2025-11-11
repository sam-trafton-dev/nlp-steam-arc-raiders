#!/usr/bin/env python3
"""
aggregate_insights.py
---------------------
Reads review_summaries.jsonl (from summarize_reviews.py)
and creates a high-level developer report.
"""
import json, subprocess, textwrap
from collections import Counter
from tqdm import tqdm

MODEL = "analyst" # or any local Ollama model

def query_ollama(prompt):
    cmd = ["ollama", "run", MODEL]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate(prompt)
    return out.strip()

def main():
    summaries = [json.loads(l) for l in open("analysis_out/review_summaries.jsonl", encoding="utf-8") if l.strip()]
    tasks = [s.get("task","").lower() for s in summaries if s.get("task")]
    likes = [s.get("like","") for s in summaries if s.get("like")]
    dislikes = [s.get("dislike","") for s in summaries if s.get("dislike")]
    confidence = [s.get("self_confidence", "") for s in summaries if s.get("self_confidence")]

    # Quick frequency list for reference
    top_tasks = Counter(tasks).most_common(20)

    prompt = textwrap.dedent(f"""
    You are analyzing summarized Steam reviews for a video game.
    Each entry contains what players like, dislike, and the dev task suggested.

    TASK LIST (most common first):
    {top_tasks}

    GOAL:
    1. Synthesize the 3–5 most critical actionable themes.
    2. Write concrete sprint objectives developers could implement.
    3. Provide one-paragraph executive summary.

    Respond in Markdown.
    """)

    print("Generating aggregate developer report...")
    result = query_ollama(prompt)

    with open("analysis_out/aggregate_report.txt", "w", encoding="utf-8") as f:
        f.write(result)

    print("\nReport written → analysis_out/aggregate_report.txt")

if __name__ == "__main__":
    main()
