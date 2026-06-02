"""RAGAS 测试集生成。"""
import logging

from langchain_core.documents import Document

logger = logging.getLogger("ragmate")


def generate_testset(
    documents: list[Document],
    size: int = 50,
    seed: int = 42,
    query_distribution: dict = None,
) -> list[dict]:
    """Generate test cases from documents using RAGAS TestsetGenerator."""
    from ragas.testset import TestsetGenerator
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from backend.infrastructure.model_factory import get_llm, get_embeddings

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    kwargs = {}
    if query_distribution:
        kwargs["query_distribution"] = query_distribution

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=size,
        **kwargs,
    )

    test_cases = []
    for sample in testset.samples:
        eval_sample = sample.eval_sample
        test_cases.append({
            "user_input": eval_sample.user_input,
            "reference": eval_sample.reference or "",
            "retrieved_contexts": [],
            "response": "",
            "source": "",
        })

    return test_cases
