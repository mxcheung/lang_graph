"""Step 2 - Task 2 — Embed one cleaned movie and inspect the resulting vector"""
import json
import sys
from pathlib import Path

from embedding import bundle_path, embed_query

CLEAN_PATH = bundle_path("movies_clean")


def embed_and_inspect(sample: dict) -> None:

    # Build the text we send to the embedding model: title + summary
    text = f"Title: {sample['title']}\n\nSummary: {sample['summary']}"
    print(f"Text being embedded:\n{text}\n")

    # User task: Convert the text to embedding vector using embed_query
    vector = embed_query(text)

    # Inspect: an embedding is just a list of numbers
    print(f"First 10 dimensions: {vector[:10]}")
    print(f"Embedding length:    {len(vector)}")


# Usage:
#   python src/embed_sample.py                       -> first movie in the dataset
#   python src/embed_sample.py "<title substring>"   -> first movie matching the substring
#
# Examples:
#   python src/embed_sample.py matrix                -> "The Matrix"
#   python src/embed_sample.py "shining"             -> "The Shining"
#   python src/embed_sample.py notebook              -> "The Notebook"
def main() -> None:

    # Load the cleaned dataset (produced by preprocess.py)
    movies = json.loads(CLEAN_PATH.read_text())

    # If a substring is given, find the first matching title; otherwise pick movies[0]
    if len(sys.argv) > 1:
        needle = sys.argv[1].lower()
        matches = [m for m in movies if needle in m["title"].lower()]
        if not matches:
            sys.exit(f"No movie matches: {sys.argv[1]!r}")
        sample = matches[0]
    else:
        sample = movies[0]

    embed_and_inspect(sample)


if __name__ == "__main__":
    main()
