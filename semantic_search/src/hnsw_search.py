"""Step 3 - Task 2 — Build a FAISS HNSW index and compare against Flat"""
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

from embedding import bundle_path, embed_texts, embedding_alias

MOVIES_BUNDLE = bundle_path("movies_clean", embedding_alias())

SAMPLE_QUERY = "movies about artificial intelligence and machines"
K = 20

# HNSW parameters
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 40
HNSW_EF_SEARCH = 32


def build_hnsw_index(
    movie_vectors: np.ndarray,
    M: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
    ef_search: int = HNSW_EF_SEARCH,
) -> faiss.IndexHNSWFlat:

    # HNSW works on (N, D) float32 arrays, the same format used by Flat indexes
    dim = movie_vectors.shape[1]

    # User task: Create an HNSW index that uses inner product similarity
    index_hnsw = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)

    # User task: Control how carefully the HNSW graph is built
    index_hnsw.hnsw.efConstruction = ef_construction

    # User task: Control how deeply the graph is searched at query time
    index_hnsw.hnsw.efSearch = ef_search

    # User task: Add vectors to the index and build the HNSW graph
    index_hnsw.add(movie_vectors)

    return index_hnsw

# Usage:
#   python src/hnsw_search.py                            -> uses SAMPLE_QUERY
#   python src/hnsw_search.py "<query>"                  -> overrides the query
#
# Examples:
#   python src/hnsw_search.py "feel-good romantic comedy"
#   python src/hnsw_search.py "scary haunted house"
def main() -> None:

    # If a CLI query is given, use it; otherwise fall back to SAMPLE_QUERY
    query = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUERY

    # Load the per-model bundle (cleaned movies + raw embeddings) and normalize for FAISS IP search
    bundle = json.loads(MOVIES_BUNDLE.read_text())
    movies = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]
    movie_vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
    faiss.normalize_L2(movie_vectors)
    dim = movie_vectors.shape[1]
    print(f"Loaded {movie_vectors.shape[0]} movies, {movie_vectors.shape[1]}-dim embeddings from {MOVIES_BUNDLE}")

    # Flat index (exact brute-force, IP = cosine for normalized vectors)
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(movie_vectors)

    # HNSW index (approximate, IP = cosine for normalized vectors)
    index_hnsw = build_hnsw_index(movie_vectors)

    # Embed the query and L2-normalize so it lives in the same space as the indexed vectors
    raw = embed_texts([query])
    query_vec = np.array(raw, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Run the same query through both indexes
    _, flat_idxs = index_flat.search(query_vec, K)
    _, hnsw_idxs = index_hnsw.search(query_vec, K)

    # Side-by-side ranking comparison
    print(f"\nQuery: {query!r}\n")
    print(f"{'Rank':<5}{'Flat (exact)':<30}{'HNSW (approx)':<30}")
    for rank in range(K):
        print(f"{rank + 1:<5}{movies[flat_idxs[0][rank]]['title']:<30}{movies[hnsw_idxs[0][rank]]['title']:<30}")


if __name__ == "__main__":
    main()
