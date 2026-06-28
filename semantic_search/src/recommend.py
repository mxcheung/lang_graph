"""Step 3 - Task 4 — Recommend similar movies"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from embedding import bundle_path, embedding_alias
from hnsw_search import build_hnsw_index

MOVIES_BUNDLE = bundle_path("movies_clean", embedding_alias())

SAMPLE_TITLES = [
    "The Matrix",
    "The Notebook",
    "The Shining",
]
K = 5


def load_index() -> tuple[list[dict], np.ndarray, dict[str, int], faiss.IndexHNSWFlat]:

    if not MOVIES_BUNDLE.exists():
        sys.exit(
            f"Bundle not found: {MOVIES_BUNDLE}\n"
            f"Run `python build/build_corpus.py ollama qwen3-embedding:latest` first."
        )

    # Load the per-model bundle (cleaned movies + raw embeddings); normalize for FAISS IP
    bundle = json.loads(MOVIES_BUNDLE.read_text())
    movies = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]
    movie_vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
    faiss.normalize_L2(movie_vectors)

    title_to_idx = {m["title"]: i for i, m in enumerate(movies)}
    print(f"Loaded {movie_vectors.shape[0]} movies, {movie_vectors.shape[1]}-dim embeddings from {MOVIES_BUNDLE}\n")

    # Reuse the HNSW builder from Task 2 — same M / efConstruction / efSearch defaults
    index = build_hnsw_index(movie_vectors)

    return movies, movie_vectors, title_to_idx, index


def resolve_titles(titles: list[str], title_to_idx: dict[str, int]) -> list[int]:

    # Convert titles to indexes; print warnings for unknown titles
    valid_idxs = [title_to_idx[t] for t in titles if t in title_to_idx]
    for t in titles:
        if t not in title_to_idx:
            print(f"Unknown title: {t!r}")

    return valid_idxs


def print_results(movies: list[dict], skip: set, neighbor_idxs, scores, header: str) -> None:

    print(header)
    rank = 1
    for n, s in zip(neighbor_idxs, scores):
        if int(n) in skip:
            continue  
        
        m = movies[n]
        print(f"  {rank}. {m['title']:30s}  {m['primary_genre']:20s}  sim={s:.3f}")
        rank += 1

    print()


def recommend_each(titles: list[str]) -> None:
    movies, movie_vectors, title_to_idx, index = load_index()

    for title in titles:
        if title not in title_to_idx:
            print(f"Unknown title: {title!r}\n")
            continue

        # Use this movie's own vector as the query; ask K+1 so we can drop the self-match
        idx = title_to_idx[title]
        query_vec = movie_vectors[idx:idx + 1]
        scores, neighbor_idxs = index.search(query_vec, K + 1)

        print_results(
            movies,
            skip={idx},
            neighbor_idxs=neighbor_idxs[0],
            scores=scores[0],
            header=f"More like: {title} ({movies[idx]['primary_genre']})",
        )


def recommend_for_many(titles: list[str]) -> None:
    movies, movie_vectors, title_to_idx, index = load_index()

    valid_idxs = resolve_titles(titles, title_to_idx)
    if not valid_idxs:
        print("No valid titles given.\n")
        return

    # User task: Calculate the mean vector of the input movies
    mean_vec = movie_vectors[valid_idxs].mean(axis=0, keepdims=True)

    # User task: Re-normalize the mean vector for cosine-style similarity using faiss.normalize_L2
    faiss.normalize_L2(mean_vec)

  
    # User task: Search for K + N neighbors
    scores, neighbor_idxs = index.search(mean_vec, K + len(valid_idxs))

    titles_str = ", ".join(movies[i]["title"] for i in valid_idxs)
    print_results(
        movies,
        skip=set(valid_idxs),
        neighbor_idxs=neighbor_idxs[0],
        scores=scores[0],
        header=f"More like: [{titles_str}] (mean of {len(valid_idxs)} movies)",
    )

# Usage:
#   python src/recommend.py                                       -> 'each' mode, SAMPLE_TITLES
#   python src/recommend.py "The Matrix"                          -> 'each' mode, one title
#   python src/recommend.py each "The Matrix" "The Notebook"      -> 'each' mode, custom titles
#   python src/recommend.py many "The Matrix" "The Notebook"      -> 'many' mode (mean centroid)
def main() -> None:
    args = sys.argv[1:]
    mode = "each"

    # Strip leading mode token if present
    if args and args[0] in {"each", "many"}:
        mode = args[0]
        args = args[1:]

    titles = args if args else SAMPLE_TITLES

    if mode == "many":
        recommend_for_many(titles)
    else:
        recommend_each(titles)


if __name__ == "__main__":
    main()
