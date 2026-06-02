# Cross-Encoder Reranking for Retrieval

## Abstract
Cross-encoder rerankers jointly encode a query and a candidate passage in a single
transformer pass, allowing full attention between query and passage tokens. They are
more accurate than bi-encoders but too expensive to run over an entire corpus, so they
are applied as a second stage over a small candidate set produced by a fast retriever.

## Method
A pretrained model such as BGE-reranker is fine-tuned to output a relevance score for
each (query, passage) pair. The first-stage retriever returns the top-k candidates,
and the cross-encoder rescores and reorders them before synthesis.

## Results
Adding cross-encoder reranking on top of dense or hybrid first-stage retrieval
consistently improves precision at the top ranks, which is critical for grounding and
citation accuracy in retrieval-augmented generation.
