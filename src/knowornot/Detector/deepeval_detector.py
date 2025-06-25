from typing import Dict, Any
from datetime import datetime
import os
import asyncio

from ..common.models import LLMResponseWithEvaluation
from .models import DetectorType, DetectionStatus, DetectionResult, DetectionLabel
from . import Detector

from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase


class DeepEvalDetector(Detector):
    """Detector using DeepEval's faithfulness metric to assess answer trustworthiness."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the DeepEval detector.

        Args:
            confidence_threshold: Threshold for detection confidence
        """
        super().__init__(
            detector_type=DetectorType.DEEPEVAL,
            confidence_threshold=confidence_threshold,
        )
        self.metric = None

    def _validate_config(self) -> None:
        """Validate that required configuration parameters are present."""
        # Check for OpenAI API key
        if "openai_api_key" not in self.config and "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "DeepEval requires an OpenAI API key. Provide 'openai_api_key' in the config or set the OPENAI_API_KEY environment variable."
            )

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the DeepEval detector.

        Args:
            config: Configuration parameters (can include 'openai_api_key')
        """
        # Set OPENAI_API_KEY environment variable if provided
        if "openai_api_key" in config:
            os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
        super().configure(config)

        # Initialize the faithfulness metric
        self.metric = FaithfulnessMetric(threshold=self.confidence_threshold)

    def detect(self, response: LLMResponseWithEvaluation) -> DetectionResult:
        """
        Detect if a response is factual using DeepEval's faithfulness metric.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult containing the detection outcome
        """

        if not self.metric:
            error_msg = (
                "DeepEval detector not configured. Please call configure() first."
            )
            self.logger.error(error_msg)
            return DetectionResult(
                detector_type=self.detector_type,
                confidence=None,
                explanation=error_msg,
                status=DetectionStatus.FAILED,
                error=error_msg,
            )

        try:
            # Extract required fields from response using the same pattern as AWS Bedrock detector
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

            # Create test case for DeepEval
            test_case = LLMTestCase(
                input=question, actual_output=model_answer, retrieval_context=context
            )

            # Run evaluation
            self.metric.measure(test_case)

            # Check if we got a valid score
            if self.metric.score is None:
                error_msg = "DeepEval failed to produce a valid score"
                self.logger.error(error_msg)
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=None,
                    explanation=error_msg,
                    status=DetectionStatus.FAILED,
                    error=error_msg,
                )

            # Determine label based on score and threshold
            if self.metric.score >= self.confidence_threshold:
                label = DetectionLabel.FACTUAL
            else:
                label = DetectionLabel.NON_FACTUAL

            return DetectionResult(
                detector_type=self.detector_type,
                label=label,
                confidence=self.metric.score,
                explanation=self.metric.reason,
                status=DetectionStatus.SUCCESS,
            )

        except Exception as e:
            error_msg = f"Error in DeepEval detection: {str(e)}"
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
        """Get metadata about the detector."""
        return {
            "detector_type": self.detector_type,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": datetime.now().isoformat(),
        }
