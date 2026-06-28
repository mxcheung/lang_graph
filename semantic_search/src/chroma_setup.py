"""Step 4 - Task 1 — Set up ChromaDB and ingest movies"""
import json
import sys
from pathlib import Path

# pysqlite3 → sqlite3 before chromadb import
import sqlite3_shim
import chromadb
import numpy as np

from embedding import bundle_path, embedding_alias

MOVIES_BUNDLE = bundle_path("movies_clean", embedding_alias())
CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "movies"

def init_client(path: Path) -> chromadb.PersistentClient:

    # PersistentClient writes the collection data to disk so re-runs see it
    return chromadb.PersistentClient(path=str(path))


def create_collection(client: chromadb.PersistentClient, name: str, recreate: bool = True) -> chromadb.Collection:

    # If recreate=True, drop the existing collection first so this is idempotent
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # User task: Create a Chroma collection that uses cosine distance ('hnsw:space': 'cosine')
    return client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:

    return client.get_collection(name)


def add_records(collection: chromadb.Collection, movies: list[dict], movie_vectors: np.ndarray) -> None:

    ids = [str(m["id"]) for m in movies]
    documents = [f"Title: {m['title']}\n\nSummary: {m['summary']}" for m in movies]
    metadatas = [
        {
            "title": m["title"],
            "year": m["year"],
            "primary_genre": m["primary_genre"],
            "genres_str": ", ".join(m["genres"]),  # Chroma metadata values must be scalar
        }
        for m in movies
    ]

    # User task: Add records to the collection
    collection.add(
        ids=ids,
        embeddings=movie_vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )    


def get_records(collection: chromadb.Collection, ids: list[str]) -> dict:
    
    return collection.get(ids=ids, include=["documents", "metadatas"])


def update_records(
    collection: chromadb.Collection,
    ids: list[str],
    metadatas: list[dict] | None = None,
    documents: list[str] | None = None,
) -> None:

    kwargs: dict = {"ids": ids}
    if metadatas is not None:
        kwargs["metadatas"] = metadatas
    if documents is not None:
        kwargs["documents"] = documents
    collection.update(**kwargs)


def delete_records(collection: chromadb.Collection, ids: list[str]) -> None:
    
    collection.delete(ids=ids)

def main() -> None:

    if not MOVIES_BUNDLE.exists():
        sys.exit(
            f"Bundle not found: {MOVIES_BUNDLE}\n"
            f"Run `python build/build_corpus.py ollama qwen3-embedding:latest` first."
        )

    # Load the per-model bundle 
    bundle = json.loads(MOVIES_BUNDLE.read_text())
    movies = [{k: v for k, v in m.items() if k != "embedding"} for m in bundle]
    movie_vectors = np.array([m["embedding"] for m in bundle], dtype=np.float32)
    print(f"Loaded {movie_vectors.shape[0]} movies, {movie_vectors.shape[1]}-dim embeddings from {MOVIES_BUNDLE}")

    # Run the Chroma ops in sequence
    client = init_client(CHROMA_PATH)
    collection = create_collection(client, COLLECTION_NAME, recreate=True)
    add_records(collection, movies, movie_vectors)

    print(f"\nCollection '{COLLECTION_NAME}' now has {collection.count()} records.")

    # Show one sample record so the stored schema is visible
    sample = get_records(collection, ids=["1"])
    print("\nSample record (id=1):")
    print(json.dumps({
        "id": sample["ids"][0],
        "metadata": sample["metadatas"][0],
        "document_preview": sample["documents"][0][:120] + "...",
    }, indent=2))


if __name__ == "__main__":
    main()
