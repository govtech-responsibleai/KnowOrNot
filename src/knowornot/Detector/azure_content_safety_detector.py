import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from . import Detector
from .models import DetectorType, DetectionResult, DetectionStatus, DetectionLabel


class AzureContentSafetyDetector(Detector):
    """
    A detector that uses Azure AI Content Safety API to assess answer trustworthiness.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the Azure Content Safety detector.

        Args:
            confidence_threshold: Threshold for detection confidence
        """
        super().__init__(
            detector_type=DetectorType.AZURE_CONTENT_SAFETY,
            confidence_threshold=confidence_threshold,
        )
        self.api_key: Optional[str] = None
        self.endpoint: Optional[str] = None
        self.region: Optional[str] = None
        self.api_version: Optional[str] = None

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the detector with Azure Content Safety credentials and settings.

        Args:
            config: Dictionary containing:
                - api_key: Azure Content Safety API key
                - endpoint: Azure Content Safety endpoint URL
                - region: Azure region
                - api_version: API version (optional, defaults to "2024-09-15-preview")
        """
        super().configure(config)

        required_fields = ["api_key", "endpoint", "region"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        self.api_key = config["api_key"]
        self.endpoint = config["endpoint"]
        self.region = config["region"]
        self.api_version = config.get("api_version", "2024-09-15-preview")

    def _validate_config(self) -> None:
        """Validate that required configuration parameters are present."""
        required_params = ["api_key", "endpoint", "region"]
        missing_params = [
            param for param in required_params if param not in self.config
        ]
        if missing_params:
            raise ValueError(
                f"Missing required configuration parameters: {', '.join(missing_params)}"
            )

    def _make_request(self, data: dict) -> Optional[float]:
        """
        Make a request to the Azure Content Safety API.

        Args:
            data: The request payload

        Returns:
            Optional[float]: The trustworthiness score (1 - ungroundedPercentage) if successful, None if failed
        """
        if not self.api_key or not self.endpoint or not self.region:
            raise ValueError("Detector not configured. Call configure() first.")

        url = f"{self.endpoint}/contentsafety/text:detectGroundedness?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        req = urllib.request.Request(
            url,
            headers=headers,
            # data=bytes(json.dumps(data).encode("utf-8")),
            data=bytes(json.dumps(data).encode("utf-8")),
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                response_data = json.loads(response.read())
                # Extract the ungrounded percentage and invert it to get trustworthiness score
                ungrounded_percentage = response_data.get("ungroundedPercentage")
                # Invert the score since higher ungrounded percentage means less trustworthy
                trustworthiness_score = 1.0 - ungrounded_percentage
                return trustworthiness_score
            else:
                raise ValueError(
                    f"API request failed with status code: {response.getcode()}"
                )

    def detect(self, response: Any) -> DetectionResult:
        """
        Run Azure Content Safety detection on a response.

        Args:
            response: The response to analyze (must have llm_response.experiment_input.source_context_qa.question,
                     llm_response.response, and llm_response.experiment_input.source_context_qa.context attributes)

        Returns:
            DetectionResult: Result containing the trustworthiness score and status
        """
        if not self.api_key or not self.endpoint or not self.region:
            error_msg = "Azure Content Safety detector not configured. Please call configure() first."
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
            data = {
                "qna": {"query": question},
                "text": model_answer,
                "groundingSources": context,
            }
            try:
                score = self._make_request(data)
                if score is None:
                    error_msg = "Failed to get trustworthiness score from Azure Content Safety API"
                    self.logger.error(error_msg)
                    return DetectionResult(
                        detector_type=self.detector_type,
                        confidence=None,
                        explanation=error_msg,
                        status=DetectionStatus.FAILED,
                        error=error_msg,
                    )
                if score >= self.confidence_threshold:
                    label = DetectionLabel.FACTUAL
                else:
                    label = DetectionLabel.NON_FACTUAL
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=score,
                    status=DetectionStatus.SUCCESS,
                    label=label,
                )
            except Exception as e:
                if isinstance(e, urllib.error.HTTPError) and e.code == 400:
                    self.logger.info(
                        "Attempting batching of Azure Content Safety requests."
                    )
                    # Batch context_questions into groups of 5
                    batch_size = 20
                    n = len(context_questions)
                    ungrounded_scores = []
                    for i in range(0, n, batch_size):
                        batch = context_questions[i : i + batch_size]
                        batch_context = [
                            "Question: " + c.question + "\nAnswer: " + c.answer
                            for c in batch
                        ]
                        batch_data = {
                            "qna": {"query": question},
                            "text": model_answer,
                            "groundingSources": batch_context,
                        }
                        try:
                            # Directly use urllib to get the ungroundedPercentage
                            url = f"{self.endpoint}/contentsafety/text:detectGroundedness?api-version={self.api_version}"
                            headers = {
                                "Content-Type": "application/json",
                                "Cache-Control": "no-cache",
                                "Ocp-Apim-Subscription-Key": self.api_key,
                            }
                            req = urllib.request.Request(
                                url,
                                headers=headers,
                                data=bytes(json.dumps(batch_data).encode("utf-8")),
                                method="POST",
                            )
                            with urllib.request.urlopen(req) as response:
                                if response.getcode() == 200:
                                    response_data = json.loads(response.read())
                                    ungrounded_percentage = response_data.get(
                                        "ungroundedPercentage"
                                    )
                                    if ungrounded_percentage is not None:
                                        ungrounded_scores.append(ungrounded_percentage)
                        except Exception as batch_e:
                            self.logger.error(
                                f"Azure Content Safety batch {i // batch_size + 1} failed: {batch_e}"
                            )
                            return DetectionResult(
                                detector_type=self.detector_type,
                                confidence=None,
                                explanation=f"{batch_e}. Attempted batching requests but Azure Content Safety batch {i // batch_size + 1} failed: {batch_e}",
                                status=DetectionStatus.FAILED,
                                error=str(batch_e),
                            )
                    if ungrounded_scores:
                        max_ungrounded = max(ungrounded_scores)
                        trustworthiness_score = 1.0 - max_ungrounded
                        if trustworthiness_score >= self.confidence_threshold:
                            label = DetectionLabel.FACTUAL
                        else:
                            label = DetectionLabel.NON_FACTUAL
                        self.logger.info(
                            "Batched requests for Azure Content Safety successful."
                        )
                        return DetectionResult(
                            detector_type=self.detector_type,
                            confidence=trustworthiness_score,
                            status=DetectionStatus.SUCCESS,
                            label=label,
                        )
                    else:
                        return DetectionResult(
                            detector_type=self.detector_type,
                            confidence=None,
                            explanation="Attempted batching requests but no valid ungrounded scores returned from batched Azure requests.",
                            status=DetectionStatus.FAILED,
                            error="No valid ungrounded scores.",
                        )
                else:
                    self.logger.error(f"Error in Azure Content Safety detection: {e}")
                    return DetectionResult(
                        detector_type=self.detector_type,
                        confidence=None,
                        explanation=f"Error in Azure Content Safety detection: {str(e)}",
                        status=DetectionStatus.FAILED,
                        error=str(e),
                    )
        except Exception as e:
            self.logger.error(f"Error in Azure Content Safety detection: {e}")
            return DetectionResult(
                detector_type=self.detector_type,
                confidence=None,
                explanation=f"Error in Azure Content Safety detection: {str(e)}",
                status=DetectionStatus.FAILED,
                error=str(e),
            )

    async def detect_async(self, response: Any) -> DetectionResult:
        """
        Asynchronous version of detect. Currently just calls the synchronous version
        since urllib.request is synchronous.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult: Result containing the trustworthiness score and status
        """
        # return self.detect(response)
        return await asyncio.to_thread(self.detect, response)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the detector."""
        return {
            "detector_type": self.detector_type,
            "confidence_threshold": self.confidence_threshold,
            "timestamp": datetime.now().isoformat(),
            "additional_config": {
                "endpoint": self.endpoint,
                "region": self.region,
                "api_version": self.api_version,
            },
        }
