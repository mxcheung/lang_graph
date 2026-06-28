"""Step 5 - Task 2 — Recall@K and Found/Missed/Snuck-in analysis"""
import json
import sys
from pathlib import Path

from chroma_search import open_collection, search_collection
from embedding import bundle_path

CLEAN_PATH = bundle_path("movies_clean")
EVAL_PATH = bundle_path("eval_queries")
KS = [5, 10]

# K used for the Found/Missed/Snuck-in breakdown
K_ANALYSIS = 5  


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:

    # User task: Calculate Recall@K
    top_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(top_k & relevant_ids) / len(relevant_ids)

    


def show_one(entry: dict, collection, movies_by_id: dict[int, dict]) -> None:

    query = entry["query"]
    relevant = set(entry["relevant_ids"])

    results = search_collection(collection, query, n_results=max(KS))
    retrieved_ids = [int(i) for i in results["ids"][0]]

    # Top-K subset for the Found/Missed/Snuck-in breakdown
    top_k_list = retrieved_ids[:K_ANALYSIS]
    top_k_set = set(top_k_list)

    found = [mid for mid in top_k_list if mid in relevant]
    missed = sorted(relevant - top_k_set)
    snuck_in = [mid for mid in top_k_list if mid not in relevant]

    print(f"Query: {query!r}")
    print(f"Relevant in eval set ({len(relevant)}): {sorted(relevant)}\n")
    for k in KS:
        print(f"R@{k:<3} = {recall_at_k(retrieved_ids, relevant, k):.2f}")
    print()

    print(f"=== Per-query analysis (K={K_ANALYSIS}) ===\n")

    print(f"Found ({len(found)}/{len(relevant)} relevant items returned in top {K_ANALYSIS}):")
    for mid in found:
        m = movies_by_id[mid]
        print(f"  + id={mid:<4} {m['title']:30s}  [primary_genre={m['primary_genre']}]")

    if missed:
        print(f"\nMissed ({len(missed)} relevant items NOT in top {K_ANALYSIS}):")
        for mid in missed:
            m = movies_by_id[mid]
            print(f"  - id={mid:<4} {m['title']:30s}  [primary_genre={m['primary_genre']}]")

    if snuck_in:
        print(f"\nSnuck in ({len(snuck_in)} irrelevant items in top {K_ANALYSIS}):")
        for mid in snuck_in:
            m = movies_by_id[mid]
            print(f"  ? id={mid:<4} {m['title']:30s}  [primary_genre={m['primary_genre']}]")


# Usage:
#   python src/recall_and_analysis.py "<substring>"     -> Found/Missed/Snuck-in for one query
#
# Examples:
#   python src/recall_and_analysis.py romantic          -> "feel-good romantic comedy"
#   python src/recall_and_analysis.py horror            -> "scary supernatural horror..."
#   python src/recall_and_analysis.py "wartime"         -> "wartime survival story"
def main() -> None:

    if not EVAL_PATH.exists():
        sys.exit(f"Eval set not found: {EVAL_PATH}")
    if not CLEAN_PATH.exists():
        sys.exit(f"Cleaned data not found: {CLEAN_PATH}")

    if len(sys.argv) < 2:
        sys.exit('Usage: python src/recall_and_analysis.py "<query substring>"')

    eval_set = json.loads(EVAL_PATH.read_text())
    collection = open_collection()

    # Substring match against eval queries (case-insensitive)
    needle = sys.argv[1].lower()
    matches = [e for e in eval_set if needle in e["query"].lower()]
    if not matches:
        sys.exit(f"No eval query matches: {sys.argv[1]!r}")
    if len(matches) > 1:
        print(f"(Multiple matches; using first: {matches[0]['query']!r})\n")

    # movies_by_id needed for the breakdown
    movies_by_id = {m["id"]: m for m in json.loads(CLEAN_PATH.read_text())}
    show_one(matches[0], collection, movies_by_id)


if __name__ == "__main__":
    main()
