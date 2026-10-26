# Hypotheses

## Part 3: Scaling (SIFT, 100K -> 300K -> 1M vectors)

**What is measured.** For each library size N (the first N SIFT vectors), the
time per query (single thread, one query at a time) of the cheapest setting
that reaches recall@10 >= 0.95, for exact search, IVF-Flat and HNSW. IVF-Flat's
number of lists grows with the data (nlist ~ sqrt(N) and 4*sqrt(N)).

**Hypothesis.** When the library grows 10x (100K -> 1M), the time per query at
recall 0.95 grows by about:

| Method   | Expected slowdown for 10x more data | Growth |
|----------|-------------------------------------|--------|
| Exact    | ~10x                                | proportional to N |
| IVF-Flat | ~3.16x (= sqrt(10))                 | proportional to sqrt(N) |
| HNSW     | ~1.5-2x                             | much slower than sqrt(N), closer to log(N) |

So the ordering of growth is: exact > IVF-Flat > HNSW.

**Why.**
- *Exact* compares the query with every stored vector, so the work is exactly
  proportional to the number of vectors.
- *IVF-Flat* compares the query with nlist centroids and then scans the nprobe
  nearest lists. With nlist ~ sqrt(N), each list holds about N / nlist ~ sqrt(N)
  vectors, and there are ~sqrt(N) centroids, so if the nprobe needed for 0.95
  stays about the same, the work grows like sqrt(N).
- *HNSW* walks a graph from neighbour to neighbour towards the query. A larger
  graph only adds a few extra hops (the upper layers let the search skip long
  distances), so the work should grow only slowly with N. It may need a
  somewhat larger efSearch at 1M to keep recall at 0.95, which is why we
  expect more than a pure log(N) increase.

**How we will judge it.** We fit the slope b of log(time per query) against
log(N) over the three sizes (time ~ N^b). The hypothesis is supported if
b(exact) ~ 1, b(IVF-Flat) ~ 0.5 and b(HNSW) is clearly below b(IVF-Flat),
i.e. the slowdown from 100K to 1M is roughly 10x, 3.16x and 1.5-2x.
