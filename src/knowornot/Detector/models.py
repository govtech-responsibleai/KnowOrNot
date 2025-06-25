from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from pathlib import Path
from ..common.models import SavedLLMResponse


class DetectorType(str, Enum):
    """Types of detection capabilities."""

    TLM = "tlm"
    AWS_BEDROCK = "aws_bedrock"
    AZURE_CONTENT_SAFETY = "azure_content_safety"
    DEEPEVAL = "deepeval"
    RAGAS = "ragas"


class DetectionStatus(str, Enum):
    """Status of a detection operation."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class DetectionLabel(str, Enum):
    """Labels for detection results."""

    FACTUAL = "FACTUAL"
    NON_FACTUAL = "NON_FACTUAL"
    UNCERTAIN = "UNCERTAIN"


class DetectionResult(BaseModel):
    """Model for a single detection result"""

    detector_type: DetectorType
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    status: DetectionStatus
    error: Optional[str] = None
    label: Optional[DetectionLabel] = None


class LLMResponseWithDetection(BaseModel):
    """Model for a single LLM response with its detection results"""

    llm_response: SavedLLMResponse
    detections: List[DetectionResult]

    def get_failed_detections(self) -> List[DetectionResult]:
        """Get list of failed detections"""
        return [d for d in self.detections if d.status == DetectionStatus.FAILED]

    def get_successful_detections(self) -> List[DetectionResult]:
        """Get list of successful detections"""
        return [d for d in self.detections if d.status == DetectionStatus.SUCCESS]

    def get_skipped_detections(self) -> List[DetectionResult]:
        """Get list of skipped detections"""
        return [d for d in self.detections if d.status == DetectionStatus.SKIPPED]


class DetectedExperimentDocument(BaseModel):
    """Document containing experiment results with detection outcomes"""

    path_to_store: Any  # This will be imported from common.models
    experiment_metadata: Dict[str, Any]
    evaluation_metadata: List[Dict[str, Any]]
    responses: List[LLMResponseWithDetection]
    detection_metadata: Dict[DetectorType, Dict[str, Any]]

    def save_to_json(self) -> None:
        """Save the document to a JSON file."""
        if not self.path_to_store.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {self.path_to_store}")

        self.path_to_store.write_text(self.model_dump_json(indent=2))
        return

    @staticmethod
    def load_from_json(path: Path | str) -> "DetectedExperimentDocument":
        """Load a document from a JSON file."""
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return DetectedExperimentDocument.model_validate_json(text)
