"""RAGAS-based evaluation for measuring RAG pipeline quality."""

import json
import logging
import time
from pathlib import Path

from src.config import EVAL_DATASET_DIR
from src.utils import StageTimer

logger = logging.getLogger(__name__)


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """Evaluate a single query-answer pair using RAGAS metrics.

    Computes faithfulness and answer relevancy (no ground truth needed).
    Returns {"faithfulness": float, "answer_relevancy": float}.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    from src.utils import get_embeddings, get_llm

    with StageTimer("ragas_eval_single"):
        try:
            llm = LangchainLLMWrapper(get_llm())
            emb = LangchainEmbeddingsWrapper(get_embeddings())

            sample = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
            dataset = EvaluationDataset.from_list([sample])

            metrics = [
                Faithfulness(llm=llm),
                ResponseRelevancy(llm=llm, embeddings=emb),
            ]

            result = evaluate(dataset=dataset, metrics=metrics)
            scores = result.to_pandas().iloc[0].to_dict()

            return {
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
            }
        except Exception:
            logger.exception("RAGAS single evaluation failed")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}


def evaluate_batch(dataset_path: str | Path) -> list[dict]:
    """Run evaluation on a dataset file.

    Expects JSON: [{"question": "...", "ground_truth": "..."}, ...]
    Runs the pipeline for each question, evaluates, and returns results.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, ContextRecall

    from src.pipeline.graph import run_pipeline
    from src.utils import get_embeddings, get_llm

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        return []

    with open(dataset_path) as f:
        test_data = json.load(f)

    if not isinstance(test_data, list) or not test_data:
        logger.error("Invalid dataset format.")
        return []

    llm = LangchainLLMWrapper(get_llm())
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    samples = []
    results_detail = []

    for item in test_data:
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        if not question:
            continue

        try:
            pipeline_result = run_pipeline(question)
            answer = pipeline_result["answer"]
            contexts = [s["text"] for s in pipeline_result.get("sources", [])]

            sample = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            }
            if ground_truth:
                sample["reference"] = ground_truth

            samples.append(sample)
            results_detail.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "num_sources": len(contexts),
            })
        except Exception:
            logger.exception("Pipeline failed for question: %s", question[:80])

    if not samples:
        return []

    with StageTimer("ragas_eval_batch"):
        try:
            dataset = EvaluationDataset.from_list(samples)

            metrics = [
                Faithfulness(llm=llm),
                ResponseRelevancy(llm=llm, embeddings=emb),
            ]
            # Add context recall if ground truth is available
            has_ground_truth = any(s.get("reference") for s in samples)
            if has_ground_truth:
                metrics.append(ContextRecall(llm=llm))

            result = evaluate(dataset=dataset, metrics=metrics)
            df = result.to_pandas()

            for i, row in df.iterrows():
                if i < len(results_detail):
                    results_detail[i]["faithfulness"] = float(row.get("faithfulness", 0.0))
                    results_detail[i]["answer_relevancy"] = float(
                        row.get("answer_relevancy", 0.0)
                    )
                    results_detail[i]["context_recall"] = float(
                        row.get("context_recall", 0.0)
                    ) if "context_recall" in row else None

        except Exception:
            logger.exception("RAGAS batch evaluation failed")

    return results_detail


def load_dataset(filename: str) -> list[dict]:
    """Load an evaluation dataset from the eval directory."""
    path = EVAL_DATASET_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_dataset(filename: str, data: list[dict]) -> Path:
    """Save an evaluation dataset to the eval directory."""
    EVAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DATASET_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
