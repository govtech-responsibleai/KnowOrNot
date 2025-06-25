from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime

from ..common.models import LLMResponseWithEvaluation
from .models import DetectorType, DetectionResult


class Detector(ABC):
    """Base class for all detectors."""

    def __init__(self, detector_type: DetectorType, confidence_threshold: float = 0.5):
        """
        Initialize the detector.

        Args:
            detector_type: The type of detector
            confidence_threshold: Threshold for detection confidence
        """
        self.logger = logging.getLogger(__name__)
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.config: Dict[str, Any] = {}
        self.logger.info(f"Initialized {detector_type} detector")

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the detector with the given parameters.

        Args:
            config: Configuration parameters for the detector
        """
        self.config = config
        self._validate_config()
        self.logger.info(
            f"Configured {self.detector_type} detector with parameters: {config}"
        )

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the detector configuration.
        Should raise ValueError if required parameters are missing or invalid.
        """
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the detector."""
        return {
            "detector_type": self.detector_type,
            "confidence_threshold": self.confidence_threshold,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

    @abstractmethod
    def detect(self, response: LLMResponseWithEvaluation) -> DetectionResult:
        """
        Detect if a response is factual or not.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult containing the detection outcome
        """
        pass

    @abstractmethod
    async def detect_async(
        self, response: LLMResponseWithEvaluation
    ) -> DetectionResult:
        """
        Asynchronously detect if a response is factual or not.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult containing the detection outcome
        """
        pass
