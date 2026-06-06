# Evaluation Methodology

---

## Ground Truth: Open RAG Benchmark

**Source:** vectara/open_ragbench on HuggingFace (CC-BY-NC-4.0)

1,000 arXiv research papers across ML, NLP, and computer science. 3,045 QA pairs authored by humans reading actual papers. 400 genuinely relevant papers and 600 hard negatives: topically adjacent papers that do not answer the question. Hard negatives make this more realistic than benchmarks where non-relevant documents are randomly sampled.

Ground truth format: `qrels.json` in BEIR/TREC standard format. Each query maps to one or more paper IDs with relevance scores. This enables computing Precision, Recall, MRR, and NDCG without manual labeling.

The project ships with 100 PDFs and a filtered `qrels_filtered.json` covering 281 queries whose relevant papers are in the 100-paper subset.

---

## IR Metrics

All metrics are computed against `qrels_filtered.json` with no LLM involved.

**MRR (Mean Reciprocal Rank):** Rank of the first relevant document per query. Score = 1/rank. Average over all queries. Primary metric when users care about getting at least one correct result quickly.

**Recall@K:** Fraction of relevant documents in the top K results. At K=5: does the system surface the answer in its top 5? Important when the downstream LLM generator needs the answer in context.

**NDCG@K:** Rewards relevant results at higher ranks logarithmically. Best single metric when both relevance and rank position matter.

**Precision@K:** Fraction of top-K results that are relevant. With one relevant document per query, the maximum achievable Precision@5 is 1/5 = 0.20. Included for completeness; MRR and Recall@K are the primary metrics for this dataset structure.

---

## LLM-as-Judge (Generation Quality)

After retrieval, top-K chunks are passed to `gpt-4o-mini` to generate a cited answer. A separate judge call scores the answer on 4 dimensions (1-5 scale each):

| Dimension | What it measures |
|-----------|-----------------|
| Relevance | Does the answer address what was asked? |
| Accuracy | Are the factual claims correct per the retrieved context? |
| Completeness | Does the answer cover all key aspects? |
| Citation Quality | Are citations correctly linked to the supporting chunks? |

The judge model is the same `llm-utils` client as generation. Pointing judge at a cheaper model is a one-line env change.

---

## Document-Level vs Chunk-Level Scoring

`qrels_filtered.json` is ground truth at document level: which arXiv paper answers each query. Retrieved chunks are deduplicated to unique document IDs before computing IR metrics. A retriever returning 5 chunks from 2 documents scores 2 doc IDs.

This matches what the system actually does: the user wants the right paper, not the right chunk within a paper.

---

## Known Limitations

**50-query variance:** At 50 queries, MRR variance is high. A 1-rank difference in one query shifts MRR by 0.02. Results are directional across the 18-cell grid, not statistically precise at the per-config level.

**100-paper subset:** The full Open RAG Benchmark has 1,000 papers. The 100-paper subset means some queries reference papers outside the index. These queries score MRR=0 regardless of retrieval quality.

**Judge score reflects corpus coverage:** Judge scores with 20 papers (2.52/5) vs 100 papers (4.53/5) measure how often the answer is available in context, not model quality.
