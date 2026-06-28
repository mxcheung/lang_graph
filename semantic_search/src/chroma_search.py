"""Step 4 - Task 2/3 — Semantic search against the Chroma collection"""
import json
import sys

from chroma_setup import CHROMA_PATH, COLLECTION_NAME, get_collection, get_records, init_client
from embedding import embed_texts


def embed_query_for_chroma(query: str) -> list[list[float]]:

    # Embed the query with the SAME provider used to build the index (else garbage results).
    return embed_texts([query])


def search_collection(collection, query: str, n_results: int = 5) -> dict:

    # Embed the query
    query_emb = embed_query_for_chroma(query)

    # User task: Query the Chroma collection
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        include=["metadatas", "distances"],
    )    

    return results


def search_with_filter(collection, query: str, where: dict, n_results: int = 5) -> dict:

    query_emb = embed_query_for_chroma(query)

    # User task: Query the Chroma collection with a 'where' filter
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results,
        where=where,
        include=["metadatas", "distances"],
    )
  
    return results


def print_search_results(results: dict, header: str) -> None:

    print(header)
    for rank, (meta, dist) in enumerate(
        zip(results["metadatas"][0], results["distances"][0]), start=1
    ):
        print(
            f"  {rank}. {meta['title']:30s}  ({meta['year']})  "
            f"{meta['primary_genre']:20s}  cos_dist={dist:.4f}"
        )
    print()


def open_collection():

    # Open the existing collection or exit with a helpful hint if it isn't there yet
    client = init_client(CHROMA_PATH)
    try:
        return get_collection(client, COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            f"Run `python src/chroma_setup.py` first."
        )


# Usage:
#   python src/chroma_search.py search "<query>"                   -> semantic search
#   python src/chroma_search.py filter "<query>" "<genre>" [year]  -> search with metadata filter
#   python src/chroma_search.py get <id>                           -> retrieve one record by id
#
# Examples:
#   python src/chroma_search.py search "movies about intelligence"
#   python src/chroma_search.py search "scary haunted house horror"
#   python src/chroma_search.py filter "AI movies" "Science Fiction"
#.  python src/chroma_search.py search "movies intelligence"
#.  python src/chroma_search.py filter "movies about intelligence" "Science Fiction"
#.  python src/chroma_search.py filter "movies about intelligence" "Science Fiction" 2010
#   python src/chroma_search.py filter "AI movies" "Science Fiction" 2010
#   python src/chroma_search.py get 1
def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit('Usage: search "<query>"  |  filter "<query>" "<genre>" [year]  |  get <id>')

    collection = open_collection()
    sub = args[0]

    if sub == "search":
        query = args[1]
        results = search_collection(collection, query)
        print_search_results(results, f"Query: {query!r}\n")

    elif sub == "filter":
        if len(args) < 3:
            sys.exit('Usage: filter "<query>" "<genre>" [min_year]')
        query, genre = args[1], args[2]

        if len(args) >= 4:
            # Two filters combined with AND: genre AND year >= min_year
            min_year = int(args[3])
            where = {"$and": [
                {"primary_genre": genre},
                {"year": {"$gte": min_year}},
            ]}
            header = f"Query: {query!r}  Filter: primary_genre={genre!r}, year>={min_year}\n"
        else:
            # Single filter: just the genre
            where = {"primary_genre": genre}
            header = f"Query: {query!r}  Filter: primary_genre={genre!r}\n"

        results = search_with_filter(collection, query, where)
        print_search_results(results, header)

    elif sub == "get":
        value = args[1]
        sample = get_records(collection, ids=[value])
        if not sample["ids"]:
            sys.exit(f"No record found with id={value!r}")
        print(json.dumps({
            "id": sample["ids"][0],
            "metadata": sample["metadatas"][0],
            "document": sample["documents"][0],
        }, indent=2))

    else:
        sys.exit(f"Unknown subcommand: {sub!r}. Use 'search', 'filter', or 'get'.")


if __name__ == "__main__":
    main()
