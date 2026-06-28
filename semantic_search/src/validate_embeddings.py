"""Step 2 - Task 3 — Validate embedding quality via nearest-neighbor inspection"""
import json
import sys
from pathlib import Path
import numpy as np

from embedding import bundle_path, embed_texts

CLEAN_PATH = bundle_path("movies_clean")

SAMPLE_TITLES = [
    "The Matrix",
    "Blade Runner",
    "Pride and Prejudice",
    "The Notebook",
    "The Shining",
]

SAMPLE_QUERIES = [
    "science fiction movies about machines",
    "love story with strong emotional themes",
    "scary haunted house horror",
    "regency era romance and class differences",
]

def print_ranked(samples: list[dict], scores: np.ndarray) -> None:

    # argsort returns ascending order; negate scores to rank by descending similarity
    order = np.argsort(-scores)
    for rank, j in enumerate(order, start=1):
        n = samples[j]
        print(f"  {rank}. {n['title']:25s}  {n['primary_genre']:20s}  sim={scores[j]:.3f}")


def embed_and_normalize(texts: list[str]) -> np.ndarray:

    # Get raw embeddings from the configured provider (list[list[float]])
    raw = embed_texts(texts)

    # User task: Convert to a (N, D) float32 numpy array
    vectors = np.array(raw, dtype=np.float32)

    # L2-normalize each row so every vector has length 1 for similarity comparison
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    return vectors


def get_movie_vectors() -> tuple[list[dict], np.ndarray]:

    # Load the cleaned dataset and pick our 5 sample movies by title
    movies_by_title = {m["title"]: m for m in json.loads(CLEAN_PATH.read_text())}
    samples = [movies_by_title[t] for t in SAMPLE_TITLES]

    # Build the embedding text for each movie (title + summary)
    texts = [f"Title: {m['title']}\n\nSummary: {m['summary']}" for m in samples]

    # Embed into normalized numpy vectors
    movie_vectors = embed_and_normalize(texts)

    return samples, movie_vectors


def show_similar_movies() -> np.ndarray:
    samples, movie_vectors = get_movie_vectors()
    print("Movie Samples:", [m["title"] for m in samples])

    print("Showing each movie's nearest neighbors (self at rank 1, sim=1.000)")

    # User task: Calculate cosine-similarity matrix between every pair of sample movies
    sim_matrix = movie_vectors @ movie_vectors.T
    
    print(sim_matrix)
    for i, m in enumerate(samples):
        print(f"\n{m['title']} ({m['primary_genre']})")
        print_ranked(samples, sim_matrix[i])

    return movie_vectors


def search_by_query(queries: list[str]) -> None:
    samples, movie_vectors = get_movie_vectors()
    print("Movie Samples:", [m["title"] for m in samples])

    # Embed the queries the same way we did the movies (normalized numpy vectors)
    query_vectors = embed_and_normalize(queries)

    # Cosine similarity between each query and each movie -> shape (n_queries, 5)
    query_sims = query_vectors @ movie_vectors.T

    print("Search by query")
    for q_idx, query in enumerate(queries):
        print(f"\nQuery: {query!r}")
        print_ranked(samples, query_sims[q_idx])


# Usage:
#   python src/validate_embeddings.py                  -> similarity mode
#   python src/validate_embeddings.py "love story with strong emotional themes"        -> query mode  (see SAMPLE_QUERIES)
def main() -> None:
    if len(sys.argv) > 1:
        search_by_query([sys.argv[1]])
    else:
        show_similar_movies()


if __name__ == "__main__":
    main()
