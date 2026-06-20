# src/monitoring/langsmith_eval.py
"""
LangSmith integration for RAG evaluation, tracing, and production monitoring.
Implements automated evaluation pipelines using LLM-as-judge and custom metrics.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Run, Example

from src.config import get_settings

logger = structlog.get_logger(__name__)


class FinancialRAGEvaluator:
    """
    Comprehensive evaluation suite for the financial RAG system using LangSmith.

    Implements:
    1. Faithfulness — Does the answer only use information from retrieved context?
    2. Relevance — Does the answer address the question?
    3. Groundedness — Are specific claims supported by source documents?
    4. Financial Accuracy — Are numerical values correct?
    5. Completeness — Does the answer cover all relevant aspects?
    """

    def __init__(self):
        settings = get_settings()
        self.client = Client(
            api_key=settings.langsmith.api_key,
            api_url=settings.langsmith.endpoint,
        )
        self.project_name = settings.langsmith.project

    async def create_evaluation_dataset(
        self,
        name: str,
        test_cases: list[dict[str, Any]],
    ) -> str:
        """
        Create a LangSmith dataset for evaluation.

        Args:
            name: Dataset name
            test_cases: List of dicts with 'question', 'expected_answer',
                       'expected_sources', and optional 'filters'
        """
        dataset = self.client.create_dataset(
            dataset_name=name,
            description=f"Financial RAG evaluation dataset - {datetime.now().isoformat()}",
        )

        for case in test_cases:
            self.client.create_example(
                inputs={"question": case["question"]},
                outputs={
                    "expected_answer": case.get("expected_answer", ""),
                    "expected_sources": case.get("expected_sources", []),
                },
                dataset_id=dataset.id,
            )

        logger.info(
            "evaluation_dataset_created",
            name=name,
            examples=len(test_cases),
        )
        return dataset.id

    def run_evaluation(
        self,
        target_fn,
        dataset_name: str,
        experiment_prefix: str = "financial-rag",
    ) -> dict:
        """
        Run evaluation against a dataset.

        Args:
            target_fn: The function to evaluate (takes question, returns answer)
            dataset_name: Name of the LangSmith dataset
            experiment_prefix: Prefix for the experiment name
        """
        evaluators = [
            self._faithfulness_evaluator,
            self._relevance_evaluator,
            self._accuracy_evaluator,
            self._completeness_evaluator,
        ]

        results = evaluate(
            target_fn,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata={
                "model": get_settings().openai.model,
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info("evaluation_complete", experiment=experiment_prefix)
        return results

    def _faithfulness_evaluator(self, run: Run, example: Example) -> EvaluationResult:
        """
        LLM-as-judge: Evaluate if the answer is faithful to retrieved context.
        Checks for hallucinations and unsupported claims.
        """
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        answer = run.outputs.get("answer", "")
        context = run.outputs.get("context", "")

        prompt = f"""Evaluate if the following answer is faithful to the provided context.
The answer should NOT contain information not found in the context.

Context: {context[:3000]}

Answer: {answer}

Score 1 if the answer is fully supported by the context.
Score 0.5 if partially supported.
Score 0 if the answer contains unsupported claims.

Respond with ONLY a number (0, 0.5, or 1) and a brief explanation.
Format: SCORE|EXPLANATION"""

        result = llm.invoke(prompt)
        parts = result.content.split("|", 1)
        score = float(parts[0].strip())
        explanation = parts[1].strip() if len(parts) > 1 else ""

        return EvaluationResult(
            key="faithfulness",
            score=score,
            comment=explanation,
        )

    def _relevance_evaluator(self, run: Run, example: Example) -> EvaluationResult:
        """Evaluate if the answer is relevant to the question."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")

        prompt = f"""Rate how well this answer addresses the question about financial data.

Question: {question}
Answer: {answer[:2000]}

Score 1 if the answer directly and completely addresses the question.
Score 0.5 if partially relevant.
Score 0 if off-topic or doesn't address the question.

Respond with ONLY: SCORE|EXPLANATION"""

        result = llm.invoke(prompt)
        parts = result.content.split("|", 1)
        score = float(parts[0].strip())
        explanation = parts[1].strip() if len(parts) > 1 else ""

        return EvaluationResult(
            key="relevance",
            score=score,
            comment=explanation,
        )

    def _accuracy_evaluator(self, run: Run, example: Example) -> EvaluationResult:
        """Evaluate numerical accuracy of financial answers."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        answer = run.outputs.get("answer", "")
        expected = example.outputs.get("expected_answer", "")

        if not expected:
            return EvaluationResult(
                key="financial_accuracy",
                score=None,
                comment="No expected answer for comparison.",
            )

        prompt = f"""Compare these financial answers for numerical accuracy.
Focus on: dollar amounts, dates, percentages, and calculations.

Expected: {expected[:1500]}
Actual: {answer[:1500]}

Score 1 if all numbers match.
Score 0.5 if most numbers match with minor differences.
Score 0 if significant numerical errors.

Respond with ONLY: SCORE|EXPLANATION"""

        result = llm.invoke(prompt)
        parts = result.content.split("|", 1)
        score = float(parts[0].strip())
        explanation = parts[1].strip() if len(parts) > 1 else ""

        return EvaluationResult(
            key="financial_accuracy",
            score=score,
            comment=explanation,
        )

    def _completeness_evaluator(self, run: Run, example: Example) -> EvaluationResult:
        """Evaluate whether the answer covers all relevant aspects."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        context = run.outputs.get("context", "")

        prompt = f"""Evaluate completeness of the financial analysis.

Question: {question}
Context available: {context[:2000]}
Answer: {answer[:2000]}

Score 1 if the answer covers all relevant aspects from the context.
Score 0.5 if it misses some aspects.
Score 0 if it significantly incomplete.

Respond with ONLY: SCORE|EXPLANATION"""

        result = llm.invoke(prompt)
        parts = result.content.split("|", 1)
        score = float(parts[0].strip())
        explanation = parts[1].strip() if len(parts) > 1 else ""

        return EvaluationResult(
            key="completeness",
            score=score,
            comment=explanation,
        )