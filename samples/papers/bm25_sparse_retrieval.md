# BM25 and Sparse Lexical Retrieval

## Abstract
BM25 is a probabilistic ranking function that scores documents by term frequency,
inverse document frequency, and document-length normalization. Despite its age, BM25
remains a strong and efficient baseline for first-stage retrieval and is widely used
to generate candidate sets for downstream neural rerankers.

## Method
For a query and a document, BM25 sums per-term contributions controlled by the
saturation parameter k1 and the length-normalization parameter b. Because scoring
relies on an inverted index, BM25 scales to very large corpora with low latency and
requires no training data.

## Discussion
Hybrid retrieval combines BM25 with dense bi-encoder scores, often via reciprocal
rank fusion, to capture both exact lexical matches and semantic similarity. This
hybrid approach typically outperforms either method alone on heterogeneous corpora.
