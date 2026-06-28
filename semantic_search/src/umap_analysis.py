"""Step 6 - Task 1 — UMAP analysis with curated categories and outliers"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="umap")
import umap  # noqa: E402

from embedding import bundle_path, embed_texts, embedding_alias

UMAP_ITEMS_BUNDLE = bundle_path("umap_items", embedding_alias())
RAW_ITEMS_PATH = Path("data/umap_items.json")
PLOT_PATH = Path("plots/umap_analysis.png")


def load_items_and_vectors(source: str) -> tuple[list[dict], np.ndarray, str]:
    """Load curated items + vectors either by embedding raw items or reading the pre-built bundle."""

    if source == "embed":
        items = json.loads(RAW_ITEMS_PATH.read_text())
        texts = [f"Title: {r['title']}\n\nSummary: {r['summary']}" for r in items]
        print(f"Embedding {len(texts)} items on the fly via configured provider...")
        vectors = np.array(embed_texts(texts), dtype=np.float32)
        return items, vectors, f"{RAW_ITEMS_PATH} (embedded on the fly)"

    if source == "json":
        bundle = json.loads(UMAP_ITEMS_BUNDLE.read_text())
        items = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]
        vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
        return items, vectors, str(UMAP_ITEMS_BUNDLE)

    raise ValueError(f"Unknown source: {source!r}. Use 'embed' or 'json'.")


def project_to_2d(vectors: np.ndarray, seed: int) -> np.ndarray:

    # User task: Configure UMAP to compare vectors with cosine distance
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.15,
        metric="cosine",
        random_state=seed,
    )

    # User task: Project the high-dimensional vectors into 2D coordinates
    return reducer.fit_transform(vectors)



# Usage:
#   python src/umap_analysis.py                       -> load pre-built bundle, seed=42
#   python src/umap_analysis.py json  [seed]          -> load pre-built bundle from data/embeddings/
#   python src/umap_analysis.py embed [seed]          -> embed raw items via configured provider
def main() -> None:

    args = sys.argv[1:]
    source = args[0] if args else "json"
    seed = int(args[1]) if len(args) > 1 else 42

    # Load the 16 curated items + their embeddings via the chosen source
    items, vectors, source_label = load_items_and_vectors(source)
    print(f"Loaded {vectors.shape} from {source_label}, using seed={seed}")

    # Project the embeddings to 2D
    coords_2d = project_to_2d(vectors, seed)

    # Create one color per category
    categories = sorted({item["category"] for item in items})
    palette = plt.cm.tab10(np.arange(len(categories)))
    color_map = dict(zip(categories, palette))

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, item in enumerate(items):
        x, y = coords_2d[i]
        category = item["category"]
        is_outlier = category == "Outlier"
        color = color_map[category]

        # Use a different marker for outliers instead of making them overly red
        ax.scatter(
            [x],
            [y],
            c=[color],
            s=170 if is_outlier else 120,
            marker="X" if is_outlier else "o",
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
        )

        title = item["title"]
        if len(title) > 25:
            title = title[:23] + "…"

        ax.annotate(
            title,
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if is_outlier else "normal",
        )

    # Build a legend with matching marker shapes
    legend_handles = []
    for category in categories:
        is_outlier = category == "Outlier"
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="X" if is_outlier else "o",
                color="w",
                markerfacecolor=color_map[category],
                markeredgecolor="black",
                markersize=10,
                label=category,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="best",
        fontsize=9,
        frameon=True,
        title="Category",
    )

    ax.set_title(
        f"UMAP Projection of {len(items)} Embedded Items\n"
        f"Similar items often appear closer together, but 2D projection is approximate (seed={seed})",
        fontsize=14,
    )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    print(f"Saved {PLOT_PATH}\n")

    print("Outliers included for visual inspection:")
    for item in items:
        if item["category"] == "Outlier":
            print(f"  - {item['title']}")


if __name__ == "__main__":
    main()
