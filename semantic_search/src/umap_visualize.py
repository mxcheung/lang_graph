"""Step 6 - Task 2 — Project the full corpus to 2D via UMAP, colored by genre"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="umap")
import umap  # noqa: E402

from embedding import bundle_path, embedding_alias

MOVIES_BUNDLE = bundle_path("movies_clean", embedding_alias())
PLOT_PATH = Path("plots/umap_movies.png")

def load_movies_data() -> tuple[list[dict], np.ndarray]:

    bundle = json.loads(MOVIES_BUNDLE.read_text())
    movies = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]
    vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
    print(f"Loaded {vectors.shape} from {MOVIES_BUNDLE}")

    return movies, vectors


def load_and_project(seed: int) -> tuple[list[dict], np.ndarray]:

    # User task: Load the movie metadata and embedding vectors
    movies, vectors = load_movies_data()

    # User task: Configure UMAP for the 100-movie corpus
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )

    # User task: Project the high-dimensional vectors into 2D and return them with the metadata
    coords_2d = reducer.fit_transform(vectors)
    return movies, coords_2d
   


# Usage:
#   python src/umap_visualize.py                  -> default seed (42)
#   python src/umap_visualize.py <seed>           -> custom random seed (UMAP is stochastic)
#
# Examples:
#   python src/umap_visualize.py
#   python src/umap_visualize.py 7
#   python src/umap_visualize.py 100
def main() -> None:

    if not MOVIES_BUNDLE.exists():
        sys.exit(
            f"Bundle not found: {MOVIES_BUNDLE}\n"
            f"Run `python build/build_corpus.py ollama qwen3-embedding:latest` first."
        )

    # Random seed: defaults to 42, optionally overridden via CLI
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    # Load the per-model bundle and project the embeddings to 2D in one step
    movies, coords_2d = load_and_project(seed)

    # Color each point by its primary genre
    genres = [m["primary_genre"] for m in movies]
    unique_genres = sorted(set(genres))
    palette = plt.cm.tab10(np.arange(len(unique_genres)))
    color_map = dict(zip(unique_genres, palette))

    fig, ax = plt.subplots(figsize=(12, 8))
    for genre in unique_genres:
        mask = np.array([g == genre for g in genres])
        pts = coords_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color_map[genre]], label=genre, s=80, alpha=0.75)

    ax.set_title(f"Movie embeddings projected to 2D via UMAP (seed={seed}, colored by primary genre)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=9)

    PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
