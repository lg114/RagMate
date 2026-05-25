"""RAGAS 指标计算。"""
import logging

from streaming_llm import ChatOpenAICompatible

logger = logging.getLogger("ragmate")


class EvalLLM(ChatOpenAICompatible):
    """Subclass that forces n=1 (some APIs like MiMo don't support n>1)."""
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("n", None)
        return payload


def compute_metrics(test_cases: list[dict], judge_model: str = None):
    """Compute RAGAS metrics on evaluated test cases."""
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from model_factory import get_llm, get_embeddings
    from streaming_llm import create_llm
    from config import settings

    if judge_model:
        base_llm = create_llm(
            model=judge_model,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE_URL,
        )
    else:
        base_llm = get_llm()

    judge_llm = LangchainLLMWrapper(EvalLLM(
        model=base_llm.model_name,
        api_key=base_llm.openai_api_key,
        base_url=base_llm.openai_api_base,
    ))

    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    try:
        from ragas.metrics import FactualCorrectness
        metrics.append(FactualCorrectness(llm=judge_llm))
    except ImportError:
        pass

    samples = [
        SingleTurnSample(
            user_input=tc["user_input"],
            response=tc["response"],
            retrieved_contexts=tc["retrieved_contexts"],
            reference=tc["reference"],
        )
        for tc in test_cases
    ]

    dataset = EvaluationDataset(samples)
    return evaluate(dataset=dataset, metrics=metrics)
