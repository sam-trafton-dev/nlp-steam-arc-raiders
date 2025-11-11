#!/usr/bin/env python3
import os, json, time, argparse, random
from typing import Dict, Any, Set
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# -------- Config ----------
load_dotenv()
BASE_URL = "https://store.steampowered.com/appreviews/{appid}"
OUT_DIR = "out_reviews"
MAX_PER_PAGE = 100
BASE_SLEEP = 0.5
MAX_RETRIES = 5
TIMEOUT = 20

def backoff_sleep(attempt: int):
    time.sleep(min(BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.3), 15))

def fetch_page(appid: int, cursor: str, language: str, offtopic: int, filter_type: str = "recent") -> Dict[str, Any]:
    """Fetch one page using Steam's storefront reviews endpoint (no day_range)."""
    params = {
        "json": 1,
        "filter": filter_type,                 # "recent" or "updated"
        "language": language,                  # "english" or "all"
        "review_type": "all",                  # all|positive|negative
        "purchase_type": "all",                # all|steam|non_steam_purchase
        "filter_offtopic_activity": offtopic,  # 1=filter review-bomb periods, 0=include
        "num_per_page": MAX_PER_PAGE,          # <= 100
        "cursor": cursor,                      # "*" first, then server-provided
    }
    url = BASE_URL.format(appid=appid)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code == 429:
                backoff_sleep(attempt); continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") != 1:
                # transient storefront hiccup; back off and retry
                backoff_sleep(attempt); continue
            return data
        except Exception:
            backoff_sleep(attempt)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts")

def fetch_all_reviews(appid: int, max_reviews: int, language: str, offtopic: int, overwrite: bool):
    os.makedirs(OUT_DIR, exist_ok=True)
    outfile = os.path.join(OUT_DIR, f"reviews_{appid}.jsonl")
    metafile = os.path.join(OUT_DIR, f"meta_{appid}.json")

    if os.path.exists(outfile) and not overwrite:
        print(f"{outfile} already exists — use --overwrite to refetch.")
        return

    cursor = "*"
    seen_cursors: Set[str] = set()
    total = 0

    with open(outfile, "w", encoding="utf-8") as f_out, tqdm(total=max_reviews, desc="Fetching reviews") as pbar:
        while total < max_reviews:
            if cursor in seen_cursors:
                print("Cursor repeated — stopping to avoid loop.")
                break
            seen_cursors.add(cursor)

            data = fetch_page(appid, cursor, language, offtopic, filter_type="recent")
            reviews = data.get("reviews", [])
            if not reviews:
                print("No more reviews returned — stopping.")
                break

            for rv in reviews:
                f_out.write(json.dumps(rv, ensure_ascii=False) + "\n")

            total += len(reviews)
            pbar.update(len(reviews))

            # Save query summary once
            if total <= MAX_PER_PAGE and data.get("query_summary"):
                with open(metafile, "w", encoding="utf-8") as m:
                    json.dump(data["query_summary"], m, indent=2)

            cursor = data.get("cursor", "")
            if not cursor or len(reviews) < MAX_PER_PAGE:
                # last page under current filters
                break

            time.sleep(BASE_SLEEP)

    print(f"\nDone. {total} reviews saved to {outfile}")

def main():
    ap = argparse.ArgumentParser(description="Fetch Steam reviews (recent feed, cursor paging)")
    ap.add_argument("--appid", type=int, required=True, help="The appid of the target game")
    ap.add_argument("--max", type=int, default=80000, help="Max reviews to fetch")
    ap.add_argument("--lang", default="english")
    ap.add_argument("--offtopic", type=int, default=1, choices=[0,1], help="1=filter review-bombs, 0=include")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing reviews output directory")
    args = ap.parse_args()

    fetch_all_reviews(args.appid, args.max, args.lang, args.offtopic, args.overwrite)

if __name__ == "__main__":
    main()
