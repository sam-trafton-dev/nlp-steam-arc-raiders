# 🎮 Arc Raiders Review Insights Dashboard

A full data pipeline and visualization tool that collects, analyzes, and interprets **Steam user reviews** for the game **Arc Raiders**, turning raw player feedback into **actionable developer insights**.

---

## 🚀 Project Overview

This project scrapes real Steam reviews, performs **sentiment analysis**, **topic extraction**, and **task categorization** using local NLP/LLM pipelines (Ollama + Mistral), then visualizes results in a **Streamlit dashboard**.

The dashboard provides:
- Player sentiment trends (positive/negative ratios, tone distribution)
- Thematic clustering of player feedback
- Automatically generated **developer sprint tasks**
- Confidence-weighted prioritization of what devs should focus on next

---

## 🧩 Features

### 🔍 Data Pipeline
- Fetches all reviews from the Steam API (`get_data.py`)
- Stores each review as JSONL for offline processing

### 🧠 NLP Analysis
- **`analyze_sentiment.py`** – Uses VADER to calculate per-review sentiment
- **`get_insights.py`** – Extracts top topics and gameplay themes (TF-IDF / RAKE / KMeans)
- **`summarize_reviews.py`** – Uses a local LLM (Mistral via Ollama) to summarize and generate actionable dev tasks
- **`aggregate_insights.py`** – Aggregates per-review tasks into developer categories

### 📊 Visualization
- **`dashboard_app.py` (Streamlit)** visualizes:
  - Sentiment histogram (color-coded negative → positive)
  - Category-weighted developer task chart
  - Confidence-based task drilldown table
  - Summary text explaining overall tone, engagement, and review variability

---

## 🎨 Design Theme

Inspired by the **Arc Raiders logo** color palette:
| Element | Color | Hex |
|----------|--------|-----|
| Negative | Red | `#F80E10` |
| Neutral | Yellow | `#F9CE08` |
| Positive | Green | `#2DF186` |
| Accent | Cyan | `#84F5ED` |

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Ysam-trafton-dev/arc-raiders-insights-dashboard.git
cd arc-raiders-insights-dashboard

streamlit run dashboard_app.py

An API key is not needed for the steam store API's

Each step can be performed independently.

