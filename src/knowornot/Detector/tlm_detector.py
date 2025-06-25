from typing import Dict, Any
from cleanlab_tlm import TLM
from datetime import datetime
import asyncio

from ..common.models import LLMResponseWithEvaluation
from .models import DetectorType, DetectionStatus, DetectionResult, DetectionLabel
from . import Detector


class TLMDetector(Detector):
    """Detector using cleanlab's TLM (Trusted Language Model) API."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the TLM detector.

        Args:
            confidence_threshold: Threshold for detection confidence
        """
        super().__init__(
            detector_type=DetectorType.TLM, confidence_threshold=confidence_threshold
        )
        self.tlm_client = None

    def _validate_config(self) -> None:
        """Validate that required configuration parameters are present."""
        if "api_key" not in self.config:
            raise ValueError(
                "TLM API key is required. Please provide it in the config."
            )

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the TLM detector.

        Args:
            config: Configuration parameters including:
                - api_key: The TLM API key
        """
        super().configure(config)
        self.tlm_client = TLM(api_key=self.config["api_key"])

    def detect(self, response: LLMResponseWithEvaluation) -> DetectionResult:
        """
        Detect if a response is factual using TLM.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult containing the detection outcome
        """
        if not self.tlm_client:
            error_msg = "TLM detector not configured. Please call configure() with an API key first."
            self.logger.error(error_msg)
            return DetectionResult(
                detector_type=self.detector_type,
                confidence=None,
                explanation=error_msg,
                status=DetectionStatus.FAILED,
                error=error_msg,
            )

        try:
            # Get trustworthiness score from TLM
            score = self.tlm_client.try_get_trustworthiness_score(
                [response.llm_response.experiment_input.source_context_qa.question],
                [response.llm_response.llm_response.response],
            )[0]["trustworthiness_score"]

            if not score:
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=None,
                    explanation="No score returned by TLM",
                    status=DetectionStatus.FAILED,
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
            error_msg = f"Error in TLM detection: {str(e)}"
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
