"""Step 2 - Task 1 — Clean the raw movie dataset"""
import json
import re
from html import unescape

from embedding import bundle_path

INPUT = bundle_path("movies")
OUTPUT = bundle_path("movies_clean")


def clean_text(text: str) -> str:

    # User task: decode HTML entities, e.g. "&#39;replicants&#39;" -> "'replicants'"
    text = unescape(text)

    # User task: strip HTML tags, e.g. "<p>In a dystopian L.A.</p>" -> " In a dystopian L.A. "
    text = re.sub(r"<[^>]+>", " ", text)

    # User task: collapse runs of whitespace and trim edges, e.g. "  hello   world  " -> "hello world"
    text = re.sub(r"\s+", " ", text).strip()


    return text


def clean_record(rec: dict) -> dict:

    # split the comma-separated genres string into a trimmed list,
    # e.g. "Science Fiction, Neo-Noir,Thriller" -> ["Science Fiction", "Neo-Noir", "Thriller"]
    genres = [g.strip() for g in rec["genres"].split(",") if g.strip()]

    return {
        "id": rec["id"],
        "title": clean_text(rec["title"]),
        "year": int(rec["year"]),
        "genres": genres,
        "primary_genre": genres[0] if genres else None,
        "summary": clean_text(rec["summary"]),
    }


def main() -> None:
    raw = json.loads(INPUT.read_text())
    cleaned = [clean_record(r) for r in raw]
    OUTPUT.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))

    print("--- BEFORE ---")
    print(json.dumps(raw[0], indent=2))
    print("\n--- AFTER ---")
    print(json.dumps(cleaned[0], indent=2, ensure_ascii=False))
    print(f"\nSaved {len(cleaned)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
