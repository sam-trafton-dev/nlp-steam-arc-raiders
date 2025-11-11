#!/usr/bin/env python3
import os, re, json, argparse, statistics
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm

INPUT_JSONL = "analysis_out/review_summaries.jsonl"
SENTIMENT_CSV = "analysis_out/sentiment_results.csv"
OUT_AGG_CSV   = "analysis_out/insights_aggregate.csv"
OUT_TASKS_CSV = "analysis_out/task_examples.csv"

# ---- normalize + bucket similar tasks into dev-focus categories ----
CATEGORIES = {
    "netcode/desync":       [r"\bdesync\b", r"netcode", r"\b(registr|hit reg)", r"packet", r"sync", r"latenc", r"rubberband", r"compensat"],
    "performance/fps":      [r"\bfps\b", r"stutter", r"frame", r"perf(ormance)?", r"optimi[sz]", r"gpu", r"cpu", r"drops?"],
    "stability/crashes":    [r"crash", r"ctd", r"freeze", r"fatal", r"hang", r"memory", r"game is not working", r"wont work"],
    "matchmaking/servers":  [r"server", r"matchmaking", r"queue", r"timeout", r"disconnect", r"dc\b"],
    "weapon/ai balance":    [r"weapon", r"gun", r"balance", r"ttk", r"time to kill", r"ai\b", r"damage", r"unbalance", r"meta"],
    "pvp experience":       [r"\bpvp\b", r"third person", r"tpv", r"camp", r"spawn", r"grief", r"toxic"],
    "pve/mission loop":     [r"\bpve\b", r"mission", r"quest", r"objective", r"loop", r"variety", r"reward", r"loot"],
    "ui/ux/controls":       [r"\bui\b", r"menu", r"hud", r"inventory", r"controls?", r"bind", r"map", r"cursor"],
    "bugs/polish":          [r"\bbug(s)?\b", r"glitch", r"polish", r"jank"],
    "anti-cheat":           [r"cheat", r"aimbot", r"wallhack", r"anti-?cheat", r"cheater"],
    "social experience": [
    # general social/co-op terms
    r"\bcoop\b", r"\bco[- ]?op\b", r"multiplayer", r"teamplay", r"team play",
    r"friends?", r"party", r"group", r"match with", r"invite", r"join (friends|party)",
    r"social", r"communication", r"chat", r"voice chat", r"mic", r"talk", r"text chat",
    # negative social issues
    r"grief", r"toxic", r"troll", r"kick(ed)?", r"report system", r"match with randoms",
    r"bad teammates?", r"team(?:mate)?s? (?:dont|won't|wont|never) (?:help|revive|play)",
    # cooperative/buddy systems
    r"buddy", r"buddies", r"ally", r"revive", r"rescue", r"support", r"assist",
    r"bounty",  # your mention – could indicate team events
]

}

def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow both one-line JSONL and pretty JSON blocks
            buf += line
            if line.endswith("}"):
                try:
                    recs.append(json.loads(buf))
                except Exception:
                    # attempt to strip ``` fences or python-isms
                    s = buf.replace("```json","").replace("```","")
                    s = re.sub(r'\bNone\b', '"None"', s)
                    try:
                        recs.append(json.loads(s))
                    except Exception:
                        pass
                buf = ""
    return recs

def categorize(task_text: str) -> str:
    if not task_text: return ""
    t = task_text.lower()
    for cat, pats in CATEGORIES.items():
        if any(re.search(p, t) for p in pats):
            return cat
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default=INPUT_JSONL)
    ap.add_argument("--in_sentiment", default=SENTIMENT_CSV)
    ap.add_argument("--out_agg", default=OUT_AGG_CSV)
    ap.add_argument("--out_tasks", default=OUT_TASKS_CSV)
    ap.add_argument("--min_conf", type=float, default=0.6, help="only count tasks at/above this confidence")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_agg), exist_ok=True)

    # Sentiment baseline (for dashboard)
    sent_df = pd.read_csv(args.in_sentiment, encoding="utf-8")
    sent_df["sentiment_bucket"] = pd.cut(sent_df["sentiment"], bins=[-1,-0.2,0.2,1], labels=["neg","mixed","pos"])

    # Tasks
    rows = load_jsonl(args.in_jsonl)
    clean = []
    for r in tqdm(rows, desc="Processing summaries"):
        review = (r.get("original_review") or r.get("review") or "").strip()
        task   = (r.get("task") or "").strip()
        conf   = r.get("confidence") or r.get("self_confidence") or 0.0
        try:
            conf = float(conf)
        except:
            conf = 0.0

        if task and task.lower() != "none" and conf >= args.min_conf:
            cat = categorize(task)
            clean.append({"category": cat, "task": task, "confidence": conf, "original_review": review})

    if not clean:
        # still write empty outputs to avoid breaking downstream steps
        pd.DataFrame(columns=["category","count","avg_confidence"]).to_csv(args.out_agg, index=False)
        pd.DataFrame(columns=["category","task","confidence","original_review"]).to_csv(args.out_tasks, index=False)
        print("No confident tasks found ≥ threshold.")
        return

    tasks_df = pd.DataFrame(clean)
    agg = (tasks_df
           .groupby("category")
           .agg(count=("task","count"), avg_confidence=("confidence","mean"))
           .reset_index()
           .sort_values(["count","avg_confidence"], ascending=[False,False]))

    # keep a few example tasks per category
    examples = (tasks_df
                .groupby("category")
                .apply(lambda g: "; ".join(g.sort_values("confidence", ascending=False)["task"].head(3)))
                .reset_index(name="examples"))

    agg = agg.merge(examples, on="category", how="left")

    agg.to_csv(args.out_agg, index=False, encoding="utf-8")
    tasks_df.to_csv(args.out_tasks, index=False, encoding="utf-8")

    # Quick CLI recommendation
    top = agg.iloc[0]
    print("\n=== Recommended Dev Focus (data-driven) ===")
    print(f"- Top category: {top['category']}  | items: {int(top['count'])}  | avg confidence: {top['avg_confidence']:.2f}")
    print(f"- Example tasks: {examples[examples['category']==top['category']]['examples'].values[0]}")
    print(f"\nWrote: {args.out_agg} and {args.out_tasks}")

if __name__ == "__main__":
    main()
