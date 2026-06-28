"""Step 3 - Task 1 — Build FAISS Flat indexes and run a sample query"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from embedding import bundle_path, embed_texts, embedding_alias

MOVIES_BUNDLE = bundle_path("movies_clean", embedding_alias())

SAMPLE_QUERY = "movies about horror and romance"

def embed_and_normalize(texts: list[str]) -> np.ndarray:

    # Get raw embeddings
    raw = embed_texts(texts)

    # Convert to a (N, D) float32 numpy array
    vectors = np.array(raw, dtype=np.float32)

    # L2-normalize each row using faiss.normalize_L2
    faiss.normalize_L2(vectors)

    return vectors


def build_flat_indexes(movie_vectors: np.ndarray) -> tuple[faiss.IndexFlatL2, faiss.IndexFlatIP]:

    # Both indexes work on (N, D) float32 arrays
    dim = movie_vectors.shape[1]

    # User task: Build IndexFlatL2 -> squared L2 distance, smaller value = closer
    index_l2 = faiss.IndexFlatL2(dim)
    index_l2.add(movie_vectors)


    # User task: Build IndexFlatIP  -> inner product, larger value = closer (= cosine for normalized vectors)
    index_ip = faiss.IndexFlatIP(dim)
    index_ip.add(movie_vectors)


    return index_l2, index_ip


def search_both_indexes(
    query: str,
    index_l2: faiss.IndexFlatL2,
    index_ip: faiss.IndexFlatIP,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Embed the query so it lives in the same space as the indexed vectors
    query_vec = embed_and_normalize([query])

  
    # User task: Run the query through the L2 index
    l2_dists, l2_idxs = index_l2.search(query_vec, k)

    # User task: Run the query through the IP index
    ip_scores, ip_idxs = index_ip.search(query_vec, k)

    return l2_dists, l2_idxs, ip_scores, ip_idxs


# Usage:
#   python src/flat_search.py                            -> uses SAMPLE_QUERY
#   python src/flat_search.py "<query>"                  -> overrides the query
#
# Examples:
#   python src/flat_search.py "movies about artificial intelligence"
#   python src/flat_search.py "scary haunted house"
def main() -> None:

    # If a CLI query is given, use it; otherwise fall back to SAMPLE_QUERY
    query = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUERY

    # Load the per-model bundle: cleaned movie records + their pre-built embeddings (raw, unnormalized)
    bundle = json.loads(MOVIES_BUNDLE.read_text())
    movies = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]

    movie_vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
    # FAISS Flat IP / L2 expect unit vectors for cosine-equivalent rankings
    faiss.normalize_L2(movie_vectors)  
    print(f"Loaded {movie_vectors.shape[0]} movies, {movie_vectors.shape[1]}-dim embeddings from {MOVIES_BUNDLE}")

    # Build both Flat indexes from the loaded vectors
    index_l2, index_ip = build_flat_indexes(movie_vectors)

    # Embed the query and search through both indexes
    l2_dists, l2_idxs, ip_scores, ip_idxs = search_both_indexes(query, index_l2, index_ip)

    print(f"\nQuery: {query!r}\n")

    print("IndexFlatL2  (smaller squared L2 distance = closer):")
    for rank, (idx, dist) in enumerate(zip(l2_idxs[0], l2_dists[0]), start=1):
        print(f"  {rank}. {movies[idx]['title']:30s}  L2_dist={dist:.4f}")

    print("\nIndexFlatIP  (larger cosine = closer):")
    for rank, (idx, score) in enumerate(zip(ip_idxs[0], ip_scores[0]), start=1):
        print(f"  {rank}. {movies[idx]['title']:30s}  cosine={score:.4f}")


if __name__ == "__main__":
    main()
