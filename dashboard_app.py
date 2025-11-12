#!/usr/bin/env python3
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from urllib.parse import quote
import json

# --- Page configuration ---
st.set_page_config(
    page_title="Arc Raiders – English Steam Review Insights Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Color palette ---
colors = {
    "negative": "#F80E10",   # red
    "neutral": "#F9CE08",    # yellow
    "positive": "#2DF186",   # green
    "accent":   "#84F5ED"    # cyan
}

# --- Custom styling ---
st.markdown(
    f"""
    <style>
        body {{
            background-color: #0E1117;
            color: #EAEAEA;
        }}
        .main-title {{
            color: {colors['accent']};
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }}
        .stMetric label, .stMetric div {{
            color: {colors['neutral']} !important;
        }}
        hr {{
            border: 1px solid {colors['accent']};
        }}
        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background-color: {colors['accent']};
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load data ---
agg_path = "analysis_out/insights_aggregate.csv"
tasks_path = "analysis_out/task_examples.csv"
sent_path = "analysis_out/sentiment_results.csv"

@st.cache_data
def load_data():
    agg = pd.read_csv(agg_path, encoding="utf-8")
    tasks = pd.read_csv(tasks_path, encoding="utf-8")
    sent = pd.read_csv(sent_path, encoding="utf-8")
    if "sentiment" in sent.columns:
        sent["bucket"] = pd.cut(sent["sentiment"], bins=[-1,-0.2,0.2,1], labels=["Negative","Neutral","Positive"])
    return agg, tasks, sent

agg, tasks, sent = load_data()

# --- Optional SVG logo ---
logo_path = Path('resources/arcRaidersFull.svg')
if logo_path.exists():
    encoded_svg = quote(logo_path.read_text(), safe="")
    st.markdown(
        f"""
        <div style='text-align:center;'>
            <img src='data:image/svg+xml;utf8,{encoded_svg}'>
            <div style='font-size:0.8rem; margin-top:4px; color:#888;'>
                Logo by <a href='https://arc-ive.net/media.html' target='_blank' style='color:#84F5ED; text-decoration:none;'>Recurrents</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div class='main-title'>English Steam Review Insights</div>", unsafe_allow_html=True)

# --- Aggregate Summary Section ---
report_path = Path("analysis_out/aggregate_report.txt")

if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        agg_report = json.load(f)

    st.markdown("### Median Review Summary")
    st.markdown(
        f"""
        This synthesized insight represents the **median player sentiment and aggregated task recommendations**  
        after analyzing **≈18 000 Arc Raiders reviews** through the full NLP and LLM pipeline(fine-tuned Mistral).
        
        
        # Overall, server synchronization issues are the most identified complaints inside negative and positive reviews.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.info("Aggregate report not found — run `aggregate_insights.py` first to generate the median review summary.")

# --- KPI metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Analyzed Reviews", f"{len(sent):,}")
with col2:
    avg_sent = sent["sentiment"].mean() if "sentiment" in sent else 0
    st.metric("Average Sentiment", f"{avg_sent:.3f}")
with col3:
    st.metric("Unique Task Categories", f"{agg['category'].nunique() if len(agg)>0 else 0}")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Top Developer Priorities ---
# Remove other category
agg_display = agg[agg["category"].str.lower() != "other"]

st.subheader("Top Developer Priorities (confidence-weighted)")
if len(agg) == 0:
    st.info("No confident tasks found. Check confidence threshold or summarization output.")
else:
    total_tasks = agg["count"].sum()
    st.caption(f"**Total actionable tasks extracted:** {total_tasks:,}")
    fig = px.bar(
        agg_display.sort_values(["count","avg_confidence"], ascending=[True, True]),
        x="count", y="category", orientation="h",
        hover_data=["avg_confidence","examples"],
        color="avg_confidence",
        color_continuous_scale=[colors["negative"], colors["neutral"], colors["positive"]],
        labels={"count":"# of Tasks", "category":"Category", "avg_confidence":"Avg Confidence"},
        title="Task Frequency by Category"
    )
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#EAEAEA"),
        coloraxis_colorbar=dict(title="Confidence")
    )
    st.plotly_chart(fig, use_container_width=True)
st.caption("Note: 'Other' category omitted for clarity (uncategorized or generic tasks).")
st.caption("Hover a bar to see representative task examples surfaced from reviews.")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")

if "bucket" in sent.columns:
    # define your red→yellow→green scale
    colorscale = [
        [0.0, colors["negative"]],
        [0.5, colors["neutral"]],
        [1.0, colors["positive"]],
    ]
    sent["bucket"] = pd.cut(
        sent["sentiment"], bins=[-1, -0.2, 0.2, 1],
        labels=["Negative", "Neutral", "Positive"]
    )

    fig_hist = px.histogram(
        sent, x="sentiment", nbins=50, color="bucket",
        color_discrete_map={
            "Negative": colors["negative"],
            "Neutral": colors["neutral"],
            "Positive": colors["positive"]
        },
        title="Sentiment Histogram (bucket-colored)"
    )
    fig_hist.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#EAEAEA"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📊 View sentiment interpretation"):
        st.markdown(
            """
            **Overall tone**  
            • Average sentiment = 0.36 (VADER scale –1 to +1) → clearly positive overall.  
            • The distribution shows a strong positive bias, consistent with a “Very Positive” Steam rating.  
            • Most players express enjoyment or satisfaction, with only a small minority showing frustration.  

            **Intensity & agreement**  
            • Std dev ≈ 0.50 → opinions are varied, meaning while the majority enjoyed the game, a vocal minority had negative experiences.  
            • The median = 0.47 confirms that most reviewers are firmly on the positive side of neutral.  
            • Not “overwhelmingly” positive, but definitely solidly favorable.  

            **Language composition**  
            • Positive = 0.31  |  Neutral = 0.61  |  Negative = 0.07  
            → Reviews contain mostly neutral language with about one-third explicitly positive tone and very little negativity.  
            • Players are giving measured, descriptive feedback rather than emotional reactions.  

            **Engagement & context**  
            • Median playtime ≈ 30 hours (≈ 1,824 minutes) before review → players are writing after substantial gameplay, not quick impressions.  
            • Up-votes ≈ 0.35  |  Funny ≈ 0.08 → community engagement is modest, typical for a relatively new or niche release.  
            • These metrics indicate authentic, experience-based reviews rather than impulsive feedback.  

            **Range of opinion**  
            • Sentiment values span the full range from –1.0 to +1.0 → small but passionate negative minority balanced by many enthusiastic supporters.  
            • The spread is healthy — indicating real diversity of opinion rather than echo-chamber positivity.  

            **Summary**  
            Player sentiment is broadly positive with moderate disagreement.  
            Most reviewers enjoy the gameplay and overall experience, while a smaller group voices specific frustrations.  
            The tone across reviews is balanced, credible, and data-backed — reflecting a genuinely well-received launch with room for improvement rather than polarized extremes.
            """
        )
    fig_pie = px.pie(
        sent, names="bucket",
        color="bucket",
        color_discrete_map={
            "Negative": colors["negative"],
            "Neutral": colors["neutral"],
            "Positive": colors["positive"]
        },
        title="Sentiment Breakdown"
    )
    fig_pie.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#EAEAEA")
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("Sentiment column not found in sentiment_results.csv")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Task Drill-down ---
st.subheader("Task Drill-down")
min_conf = st.slider("Min confidence", 0.0, 1.0, 0.6, 0.05)
cat = st.multiselect("Filter categories", sorted(tasks["category"].dropna().unique().tolist()))
df = tasks.copy()
df = df[df["category"].str.lower() != "other"]
df = df[df["confidence"] >= min_conf]
if cat:
    df = df[df["category"].isin(cat)]
st.dataframe(
    df[["category","confidence","task","original_review"]].sort_values("confidence", ascending=False),
    use_container_width=True
)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:{colors['accent']};font-weight:bold;'>— Thanks for Clicking! —</div>",
    unsafe_allow_html=True
)
