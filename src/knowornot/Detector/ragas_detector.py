from typing import Dict, Any
from datetime import datetime
import os
import asyncio

from ..common.models import LLMResponseWithEvaluation
from .models import DetectorType, DetectionStatus, DetectionResult, DetectionLabel
from . import Detector

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI


class RagasDetector(Detector):
    """Detector using Ragas Faithfulness metric to assess answer trustworthiness."""

    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__(
            detector_type=DetectorType.RAGAS, confidence_threshold=confidence_threshold
        )
        self.metric = None
        self.llm_config = None

    def _validate_config(self) -> None:
        required = [
            "azure_api_key",
            "azure_endpoint",
            "azure_api_version",
            "azure_deployment",
            "azure_model",
        ]
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(
                f"Missing required Azure LLM config for Ragas: {', '.join(missing)}"
            )

    def configure(self, config: Dict[str, Any]) -> None:
        super().configure(config)
        self.llm_config = {
            "azure_deployment": config["azure_deployment"],
            "openai_api_version": config["azure_api_version"],
            "azure_endpoint": config["azure_endpoint"],
            "model": config["azure_model"],  # optional, but good for tracing
            "validate_base_url": False,
        }
        os.environ["AZURE_OPENAI_API_KEY"] = config["azure_api_key"]
        azure_llm = AzureChatOpenAI(**self.llm_config)
        evaluator_llm = LangchainLLMWrapper(azure_llm)
        self.metric = Faithfulness(llm=evaluator_llm)

    def detect(self, response: LLMResponseWithEvaluation) -> DetectionResult:
        if not self.metric:
            error_msg = "Ragas detector not configured. Please call configure() first."
            self.logger.error(error_msg)
            return DetectionResult(
                detector_type=self.detector_type,
                confidence=None,
                explanation=error_msg,
                status=DetectionStatus.FAILED,
                error=error_msg,
            )
        try:
            question = response.llm_response.experiment_input.source_context_qa.question
            model_answer = response.llm_response.llm_response.response

            context_questions = response.llm_response.experiment_input.source_context_qa.context_questions
            if not context_questions:
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=None,
                    explanation="Skipped: No context questions found.",
                    status=DetectionStatus.SKIPPED,
                )

            context = [
                "Question: " + c.question + "\nAnswer: " + c.answer
                for c in context_questions
            ]
            sample = SingleTurnSample(
                user_input=question, response=model_answer, retrieved_contexts=context
            )
            # Faithfulness returns a score between 0 and 1
            score = self.metric.single_turn_score(sample)
            if score is None:
                error_msg = "Ragas failed to produce a valid score"
                self.logger.error(error_msg)
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=None,
                    explanation=error_msg,
                    status=DetectionStatus.FAILED,
                    error=error_msg,
                )
            # Determine label based on score and threshold
            if score >= self.confidence_threshold:
                label = DetectionLabel.FACTUAL
            else:
                label = DetectionLabel.NON_FACTUAL
            return DetectionResult(
                detector_type=self.detector_type,
                label=label,
                confidence=score,
                status=DetectionStatus.SUCCESS,
            )
        except Exception as e:
            error_msg = f"Error in Ragas detection: {str(e)}"
            self.logger.error(error_msg)
            return DetectionResult(
                detector_type=self.detector_type,
                confidence=None,
                explanation=error_msg,
                status=DetectionStatus.FAILED,
                error=str(e),
            )

    async def detect_async(self, response):
        return await asyncio.to_thread(self.detect, response)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "detector_type": self.detector_type,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": datetime.now().isoformat(),
            "additional_config": self.llm_config,
        }
