#!/usr/bin/env python3
import os, json, time
from tqdm import tqdm
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "analyst"  # any local model in Ollama

def query_ollama(prompt, timeout=90):
    cmd = ["ollama", "run", MODEL]
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",     # force UTF-8 decoding
        errors="replace"      # avoid crashes on bad bytes
    )
    try:
        out, err = p.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return '{"error": "timeout"}'
    return out.strip()

def summarize_review(text):
    prompt = f"""
You are analyzing player reviews. Follow the rules strictly.

HERE IS THE REVIEW:
{text}
END REVIEW.

Extract structured insights and return valid JSON with these keys:
- original_review: the review text
- summary: one-sentence summary of the opinion
- likes: what the player liked most
- dislikes: what the player disliked most
- task: specific technical or design task if explicitly mentioned, else "None"
- confidence: a number from 0.0 to 1.0 showing how confident you are that the "task" field is correct, 
  based only on explicit evidence in the review (1.0 = fully clear, 0.0 = pure guess)
  
  When identifying the "task" field:
- If the review directly mentions a technical or gameplay issue (e.g., desync, lag, crashes, unbalanced weapons),
  infer the most relevant and specific developer action that would resolve that issue
  (e.g., "optimize server synchronization" or "rebalance weapon damage curves").
- If the review expresses only vague dissatisfaction with no identifiable issue, set task="None".
- Do NOT invent tasks unrelated to concrete problems.

Rules:
- Never infer a task that is not clearly described.
- If no task is mentioned, set task="None" and confidence=0.0.
- Do NOT include markdown, code fences, or extra commentary.

IMPORTANT: Some reviews do not have actionable data to pull from. It is okay to set the task value to an empty string if there are no insights.
However, an empty string should not be the default. Think really hard about whether or not the player's review suggests a development task to complete.

Now return the JSON object (nothing else). Begin immediately after the marker <JSON>:
<JSON>
"""
    return query_ollama(prompt)

import re, json

def to_one_line_json(raw: str) -> str:
    # strip fences / junk
    s = raw.strip().replace("```json", "").replace("```", "")
    # if you used a sentinel like <JSON>, keep only what follows it
    if "<JSON>" in s:
        s = s.split("<JSON>", 1)[1].strip()
    # grab the first {...} block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        # fallback to a minimal error record
        return json.dumps({"error": "no_json_found", "raw": s[:200]})
    s = m.group(0)

    # common fixes: bare None/True/False → JSON-legal
    s = re.sub(r'\bNone\b', '"None"', s)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)

    # load → dump compact (one line)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return json.dumps({"error": "decode_error", "raw": s[:200]})
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main():
    df = pd.read_csv("analysis_out/sentiment_results.csv")
    reviews = df["review"].dropna().tolist()  # or slice for testing

    out_path = "analysis_out/review_summaries.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f_out, \
         ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(summarize_review, r): r for r in reviews}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            try:
                raw = future.result()
            except Exception as e:
                raw = json.dumps({"error": f"exception:{e}"})
            one_line = to_one_line_json(raw)
            f_out.write(one_line + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()
