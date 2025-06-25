from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import asyncio

from ..common.models import LLMResponseWithEvaluation
from .models import DetectorType, DetectionStatus, DetectionResult, DetectionLabel
from . import Detector


class AWSBedrockDetector(Detector):
    """Detector using AWS Bedrock's guardrails to assess answer trustworthiness."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the AWS Bedrock detector.

        Args:
            confidence_threshold: Threshold for detection confidence
        """
        super().__init__(
            detector_type=DetectorType.AWS_BEDROCK,
            confidence_threshold=confidence_threshold,
        )
        self.bedrock_client = None
        self.guardrail_identifier = None
        self.guardrail_version = None

    def _validate_config(self) -> None:
        """Validate that required configuration parameters are present."""
        required_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_region",
            "guardrail_identifier",
            "guardrail_version",
        ]
        missing_params = [
            param for param in required_params if param not in self.config
        ]
        if missing_params:
            raise ValueError(
                f"Missing required configuration parameters: {', '.join(missing_params)}"
            )

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the AWS Bedrock detector.

        Args:
            config: Configuration parameters including:
                - aws_access_key_id: AWS access key ID
                - aws_secret_access_key: AWS secret access key
                - aws_region: AWS region
                - guardrail_identifier: Bedrock guardrail identifier
                - guardrail_version: Bedrock guardrail version
        """
        super().configure(config)

        # Initialize AWS Bedrock client
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
            region_name=self.config["aws_region"],
        )

        self.guardrail_identifier = self.config["guardrail_identifier"]
        self.guardrail_version = self.config["guardrail_version"]

    def _extract_grounding_score(self, response: Dict[str, Any]) -> Optional[float]:
        """
        Extract the grounding score from the AWS Bedrock response.

        Args:
            response: The response from apply_guardrail

        Returns:
            Optional[float]: The grounding score if found, None if not found
        """
        try:
            # Get the first assessment
            assessment = response.get("assessments", [{}])[0]
            # Get the contextual grounding policy
            policy = assessment.get("contextualGroundingPolicy", {})
            # Get the filters
            filters = policy.get("filters", [])
            # Find the filter with type 'GROUNDING'
            grounding_filter = next(
                (f for f in filters if f.get("type") == "GROUNDING"), None
            )

            return grounding_filter.get("score") if grounding_filter else None

        except Exception as e:
            self.logger.error(f"Error extracting grounding score: {e}")
            return None

    def detect(self, response: LLMResponseWithEvaluation) -> DetectionResult:
        """
        Detect if a response is factual using AWS Bedrock guardrails.

        Args:
            response: The response to analyze

        Returns:
            DetectionResult containing the detection outcome
        """
        if not self.bedrock_client:
            error_msg = (
                "AWS Bedrock detector not configured. Please call configure() first."
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
            # Prepare content for guardrail evaluation
            content = [
                {
                    "text": {
                        "text": response.llm_response.llm_response.response,
                        "qualifiers": ["guard_content"],
                    }
                },
                {
                    "text": {
                        "text": response.llm_response.experiment_input.source_context_qa.question,
                        "qualifiers": ["query"],
                    }
                },
            ]
            context_questions = response.llm_response.experiment_input.source_context_qa.context_questions
            if not context_questions:
                return DetectionResult(
                    detector_type=self.detector_type,
                    confidence=None,
                    explanation="Skipped: No context questions found.",
                    status=DetectionStatus.SKIPPED,
                )

            for c in context_questions:
                content.append(
                    {
                        "text": {
                            "text": "Question: " + c.question + "\nAnswer: " + c.answer,
                            "qualifiers": ["grounding_source"],
                        }
                    }
                )

            try:
                bedrock_response = self.bedrock_client.apply_guardrail(
                    guardrailIdentifier=self.guardrail_identifier,
                    guardrailVersion=self.guardrail_version,
                    source="OUTPUT",
                    content=content,
                )
                score = self._extract_grounding_score(bedrock_response)
                if score is None:
                    error_msg = (
                        "Failed to extract grounding score from Bedrock response"
                    )
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
            except ClientError as e:
                error_msg = f"AWS Bedrock API error: {str(e)}"
                self.logger.error(error_msg)
                # Check for throttling/contextual grounding filter policy error
                if "Too many requests sent to ApplyGuardrail" in str(e):
                    self.logger.info(
                        "Attempting batching of Bedrock API Guardrail requests."
                    )
                    # Batch context_questions into groups of 5
                    batch_size = 20
                    n = len(context_questions)
                    scores = []
                    for i in range(0, n, batch_size):
                        batch = context_questions[i : i + batch_size]
                        batch_content = content[:2]  # guard_content and query
                        for c in batch:
                            batch_content.append(
                                {
                                    "text": {
                                        "text": "Question: "
                                        + c.question
                                        + "\nAnswer: "
                                        + c.answer,
                                        "qualifiers": ["grounding_source"],
                                    }
                                }
                            )
                        try:
                            batch_response = self.bedrock_client.apply_guardrail(
                                guardrailIdentifier=self.guardrail_identifier,
                                guardrailVersion=self.guardrail_version,
                                source="OUTPUT",
                                content=batch_content,
                            )
                            batch_score = self._extract_grounding_score(batch_response)
                            if batch_score is not None:
                                scores.append(batch_score)
                        except Exception as batch_e:
                            self.logger.error(
                                f"Batch {i // batch_size + 1} failed: {batch_e}"
                            )
                            return DetectionResult(
                                detector_type=self.detector_type,
                                confidence=None,
                                explanation=f"{error_msg}. Attempted batching requests but batch {i // batch_size + 1} failed: {batch_e}",
                                status=DetectionStatus.FAILED,
                                error=str(batch_e),
                            )
                    if scores:
                        min_score = min(scores)
                        if min_score >= self.confidence_threshold:
                            label = DetectionLabel.FACTUAL
                        else:
                            label = DetectionLabel.NON_FACTUAL
                        self.logger.info(
                            "Batched requests for AWS Bedrock API Guardrails successful."
                        )
                        return DetectionResult(
                            detector_type=self.detector_type,
                            label=label,
                            confidence=min_score,
                            status=DetectionStatus.SUCCESS,
                        )
                    else:
                        return DetectionResult(
                            detector_type=self.detector_type,
                            confidence=None,
                            explanation=f"{error_msg}. Attempted batching requests but no valid scores returned from batched Bedrock requests.",
                            status=DetectionStatus.FAILED,
                            error=error_msg,
                        )
                else:
                    return DetectionResult(
                        detector_type=self.detector_type,
                        confidence=None,
                        explanation=error_msg,
                        status=DetectionStatus.FAILED,
                        error=str(e),
                    )
        except Exception as e:
            error_msg = f"Error in AWS Bedrock detection: {str(e)}"
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
            "additional_config": {
                "guardrail_identifier": self.guardrail_identifier,
                "guardrail_version": self.guardrail_version,
                "aws_region": self.config.get("aws_region"),
            },
        }
