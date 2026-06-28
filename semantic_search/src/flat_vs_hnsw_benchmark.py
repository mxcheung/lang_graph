"""Step 3 - Task 3 — Benchmark FAISS Flat vs HNSW at scale to see the speed tradeoff"""
import time
import numpy as np
import faiss


def make_vectors(num_vectors, dim=384, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(num_vectors, dim)).astype(np.float32)

    # L2-normalize each row of vectors using faiss.normalize_L2
    faiss.normalize_L2(vectors)

    return vectors


def build_flat(vectors):
    dim = vectors.shape[1]
    start = time.perf_counter()

    index = faiss.IndexFlatIP(dim)       
    index.add(vectors)

    return index, time.perf_counter() - start


def build_hnsw(vectors, m=32, ef_construction=80, ef_search=64):
    dim = vectors.shape[1]
    start = time.perf_counter()

    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    index.add(vectors) 
    index.hnsw.efSearch = ef_search

    return index, time.perf_counter() - start


def time_search(index, queries, k=10, runs=5):
    index.search(queries, k)             # warmup run (not measured)

    start = time.perf_counter()
    for _ in range(runs):
        index.search(queries, k)
    avg = (time.perf_counter() - start) / runs

    return avg / len(queries) * 1000     # milliseconds per query


def main():
    DIM = 384
    K = 10
    SIZES = [10_000, 50_000]

    queries = make_vectors(200, DIM, seed=1)

    header = (f"{'Vectors':>9} | {'Flat build (s)':>14} | {'HNSW build (s)':>14} | "
              f"{'Flat search (ms)':>16} | {'HNSW search (ms)':>16} | {'Speedup':>8}")
    print("Build = time to create the index   Search = avg time for ONE query\n")
    print(header)
    print("-" * len(header))

    for n in SIZES:
        db = make_vectors(n, DIM)

        # User task: Build both indexes by calling the build_flat and build_hnsw functions
        flat, flat_build = build_flat(db)
        hnsw, hnsw_build = build_hnsw(db)

        flat_ms = time_search(flat, queries, K)
        hnsw_ms = time_search(hnsw, queries, K)

        speedup = f"{flat_ms / hnsw_ms:.1f}x"
        print(f"{n:>9,} | {flat_build:>14.3f} | {hnsw_build:>14.3f} | "
              f"{flat_ms:>16.3f} | {hnsw_ms:>16.3f} | {speedup:>8}")


if __name__ == "__main__":
    main()
