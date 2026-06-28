"""Step 5 - Task 3 — Compare precision/recall across embedding models"""
import json
from pathlib import Path

import numpy as np

from chroma_setup import CHROMA_PATH, COLLECTION_NAME, get_collection, init_client
from embedding import bundle_path, embed_texts, provider_name
from precision import precision_at_k
from recall_and_analysis import recall_at_k

EVAL_PATH = bundle_path("eval_queries")
K = 5

COMPARE_COLLECTION = "movies_all_minilm"
COMPARE_ALIAS = "all_minilm"
COMPARE_BUNDLE = bundle_path("eval_queries", COMPARE_ALIAS)


def evaluate_collection(collection_name: str, query_embeddings: list[list[float]], eval_set: list[dict]) -> list[dict]:

    # Open the pre-built Chroma collection
    client = init_client(CHROMA_PATH)
    collection = get_collection(client, collection_name)

    # Query the collection once per eval entry, then score each retrieval
    results = []
    for i, entry in enumerate(eval_set):
        relevant = set(entry["relevant_ids"])
        chroma_result = collection.query(
            query_embeddings=[query_embeddings[i]],
            n_results=K,
            include=["metadatas"],
        )
        retrieved_ids = [int(rid) for rid in chroma_result["ids"][0]]

        # User task: Append a per-query record with precision and recall scores
        results.append({
            "p": precision_at_k(retrieved_ids, relevant, K),
            "r": recall_at_k(retrieved_ids, relevant, K),
        })        
        
    return results


def show_all(variants: dict[str, list[dict]], dims: dict[str, int]) -> None:

    width = max(len(label) for label in variants)
    print(f"\n{'Collection':<{width}}  {'Dim':<8}{'P@5':<8}{'R@5':<8}")
    print("-" * (width + 26))
    for label, results in variants.items():
        mean_p = float(np.mean([r["p"] for r in results]))
        mean_r = float(np.mean([r["r"] for r in results]))
        print(f"{label:<{width}}  {dims[label]:<8}{mean_p:<8.2f}{mean_r:<8.2f}")


# Usage:
#   python src/eval_embeddings.py     -> compare P@5 / R@5 for 'movies' vs 'movies_all_minilm'
def main() -> None:
    eval_set = json.loads(EVAL_PATH.read_text())
    queries = [e["query"] for e in eval_set]

    # Embed search queries for the default collection using the configured provider
    print(f"Embedding search queries for {COLLECTION_NAME!r} (configured provider: {provider_name()})...")
    default_embeddings = embed_texts(queries)

    # Load pre-built query embeddings for the comparison collection
    print(f"Loading pre-built search query embeddings for {COMPARE_COLLECTION!r} from {COMPARE_BUNDLE}...")
    compare_embeddings = [e["embedding"] for e in json.loads(COMPARE_BUNDLE.read_text())]

    variants = {
        COLLECTION_NAME:     evaluate_collection(COLLECTION_NAME, default_embeddings, eval_set),
        COMPARE_COLLECTION:  evaluate_collection(COMPARE_COLLECTION, compare_embeddings, eval_set),
    }

    dims = {
        COLLECTION_NAME:     len(default_embeddings[0]),
        COMPARE_COLLECTION:  len(compare_embeddings[0]),
    }

    show_all(variants, dims)


if __name__ == "__main__":
    main()
