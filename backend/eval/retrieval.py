"""评估用检索包装。"""


def retrieve_contexts(question: str, top_k: int = None) -> list[str]:
    """Retrieve relevant document chunks for a question.

    Uses the full RagMate retrieval pipeline: hybrid search (dense+sparse)
    → cross-encoder reranking → score filtering → source dedup.
    """
    from retriever import retrieve
    from config import settings

    k = top_k or settings.FINAL_CONTEXT_K
    results = retrieve(question, k=k)
    return [r["text"] for r in results]
