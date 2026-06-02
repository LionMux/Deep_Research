# Dense Passage Retrieval for Open-Domain Question Answering

## Abstract
Dense Passage Retrieval (DPR) learns dense vector representations of questions and
passages using a dual-encoder architecture trained with a contrastive objective.
On open-domain question answering benchmarks, DPR retrieves relevant passages far
more accurately than sparse term-matching methods such as BM25, improving top-20
retrieval accuracy by 9–19% and yielding new state-of-the-art end-to-end QA results.

## Method
Two independent BERT encoders map questions and passages into a shared embedding
space. Relevance is scored by the inner product of the two embeddings. Training uses
in-batch negatives so that each question is contrasted against the gold passage and
the passages paired with other questions in the same mini-batch.

## Results
DPR reaches 41.5% top-20 accuracy improvement over BM25 on Natural Questions and
demonstrates that learned dense representations can replace or complement traditional
inverted-index retrieval in retrieval-augmented generation systems.
