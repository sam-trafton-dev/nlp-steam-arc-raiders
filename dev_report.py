#!/usr/bin/env python3
import pandas as pd
from textwrap import dedent

agg = pd.read_csv("analysis_out/insights_aggregate.csv", encoding="utf-8")
sent = pd.read_csv("analysis_out/sentiment_results.csv", encoding="utf-8")

total = len(sent)
avg = sent["sentiment"].mean() if "sentiment" in sent else 0
pos = (sent["sentiment"] > 0.2).mean()*100 if "sentiment" in sent else 0
neg = (sent["sentiment"] < -0.2).mean()*100 if "sentiment" in sent else 0

agg_sorted = agg.sort_values(["count","avg_confidence"], ascending=[False,False]).head(5)

lines = ["# Arc Raiders – Developer Report\n"]
lines.append(f"- Total reviews analyzed: **{total:,}**")
lines.append(f"- Average sentiment: **{avg:.3f}**  | Positive share (>0.2): **{pos:.1f}%**  | Negative share (<-0.2): **{neg:.1f}%**\n")

if len(agg_sorted):
    lines.append("## Top Priorities (confidence-weighted)")
    for _, r in agg_sorted.iterrows():
        lines.append(f"- **{r['category']}** — {int(r['count'])} items (avg conf {r['avg_confidence']:.2f})")
        if isinstance(r.get("examples",""), str) and r["examples"]:
            lines.append(f"  - Examples: {r['examples']}")
else:
    lines.append("_No confident tasks found at current threshold._")

out = "\n".join(lines) + "\n"
with open("analysis_out/dev_report.md","w",encoding="utf-8") as f:
    f.write(out)

print("Wrote analysis_out/dev_report.md")
