"""Step 5 - Task 1 — Compute precision@K for a single query"""
import json
import sys
from pathlib import Path

from chroma_search import open_collection, search_collection
from embedding import bundle_path

EVAL_PATH = bundle_path("eval_queries")
KS = [5, 10]


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:

    # User task: Calculate Precision@K
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k
    

def show_one(entry: dict, collection) -> None:

    query = entry["query"]
    relevant = set(entry["relevant_ids"])

    results = search_collection(collection, query, n_results=max(KS))
    retrieved_ids = [int(i) for i in results["ids"][0]]
    metadatas = results["metadatas"][0]

    print(f"Query: {query!r}")
    print(f"Relevant items in eval set: {sorted(relevant)}\n")

    print(f"Top {max(KS)} retrieved:")
    for rank, (rid, meta) in enumerate(zip(retrieved_ids, metadatas), start=1):
        marker = "✓ relevant" if rid in relevant else " "
        print(f"  {rank:>2}. id={rid:<4} {meta['title']:30s}  {marker}")

    print()
    for k in KS:
        p = precision_at_k(retrieved_ids, relevant, k)
        print(f"P@{k:<3} = {p:.2f}")


# Usage:
#   python src/precision.py "<substring>"           -> detailed breakdown for one eval query
#
# Examples:
#   python src/precision.py romantic                -> "feel-good romantic comedy"
#   python src/precision.py "wartime"               -> "wartime survival story"
#   python src/precision.py horror                  -> "scary supernatural horror..."
def main() -> None:

    if not EVAL_PATH.exists():
        sys.exit(f"Eval set not found: {EVAL_PATH}")

    if len(sys.argv) < 2:
        sys.exit('Usage: python src/precision.py "<query substring>"')

    eval_set = json.loads(EVAL_PATH.read_text())
    collection = open_collection()

    # Substring match against the eval-set queries (case-insensitive)
    needle = sys.argv[1].lower()
    matches = [e for e in eval_set if needle in e["query"].lower()]
    if not matches:
        sys.exit(f"No eval query matches: {sys.argv[1]!r}")
    if len(matches) > 1:
        print(f"(Multiple matches; using first: {matches[0]['query']!r})\n")

    show_one(matches[0], collection)


if __name__ == "__main__":
    main()
