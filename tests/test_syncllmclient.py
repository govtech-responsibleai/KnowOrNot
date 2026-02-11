import unittest
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from src.knowornot.config import LLMClientConfig
from src.knowornot.SyncLLMClient import SyncLLMClient, SyncLLMClientEnum, T, Message


# Define a dummy config class for testing
class DummyLLMClientConfig(LLMClientConfig):
    can_use_instructor: bool = False
    api_key: str = "dummy_api_key"  # Required for LLMClientConfig
    default_model: str = "gpt-4"


class WeatherResponse(BaseModel):
    temperature: float
    condition: str


class TestSyncLLMClient(unittest.TestCase):
    """Tests for the SyncLLMClient abstract base class."""

    def test_get_structured_response_raises_value_error_when_instructor_disabled(self):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self,
                prompt: Union[str, List[Message]],
                response_model: Type[T],
                model_used: str,
            ) -> T:
                # Mock implementation - shouldn't be called in this test
                raise NotImplementedError(
                    "_generate_structured_response should not be called"
                )

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

            def get_embedding(
                self, prompt_list: List[str], model: Optional[str] = None
            ) -> List[List[float]]:
                raise NotImplementedError(
                    "get_embedding method must be implemented in the child class"
                )

        config = DummyLLMClientConfig(
            can_use_instructor=False,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )  # Instructor disabled
        client = MockSyncLLMClient(config=config)

        with self.assertRaisesRegex(
            ValueError, "This LLM client cannot generate structured responses"
        ):
            client.get_structured_response("dummy prompt", WeatherResponse, "gpt-4")

    def test_get_structured_response_calls_generate_structured_response_when_instructor_enabled(
        self,
    ):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self,
                prompt: Union[str, List[Message]],
                response_model: Type[T],
                model_used: str,
            ) -> T:
                data: Dict[str, Any] = {"temperature": 20.0, "condition": "Cloudy"}
                instance = response_model(**data)
                return instance

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

            def get_embedding(
                self, prompt_list: List[str], model: Optional[str] = None
            ) -> List[List[float]]:
                raise NotImplementedError(
                    "get_embedding method must be implemented in the child class"
                )

        config = DummyLLMClientConfig(
            can_use_instructor=True,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )  # Instructor enabled
        client = MockSyncLLMClient(config=config)

        # Call the method with the additional ai_model parameter
        response = client.get_structured_response(
            "dummy prompt", WeatherResponse, "gpt-4"
        )

        # Assert that the response is an instance of WeatherResponse
        self.assertIsInstance(response, WeatherResponse)
        self.assertEqual(response.temperature, 20.0)
        self.assertEqual(response.condition, "Cloudy")

    @patch(
        "src.knowornot.SyncLLMClient.SyncLLMClient._generate_structured_response"
    )  # Patching the method
    def test_get_structured_response_handles_exceptions_from_generate_structured_response(
        self, mock_generate
    ):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self,
                prompt: Union[str, List[Message]],
                response_model: Type[T],
                model_used: str,
            ) -> T:
                # Mock implementation - raise an exception
                raise ValueError("Simulated error in _generate_structured_response")

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

            def get_embedding(
                self, prompt_list: List[str], model: Optional[str] = None
            ) -> List[List[float]]:
                # Default implementation - raises NotImplementedError
                raise NotImplementedError(
                    "get_embedding method must be implemented in the child class"
                )

        config = DummyLLMClientConfig(
            can_use_instructor=True,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )  # Instructor enabled
        client = MockSyncLLMClient(config=config)

        with self.assertRaisesRegex(
            ValueError, "Simulated error in _generate_structured_response"
        ):  # Expect ValueError
            client.get_structured_response("dummy prompt", WeatherResponse, "gpt-4")

    def test_prompt_abstract(self):
        # Test that inheriting classes MUST implement prompt method
        class BadClient(SyncLLMClient):
            # Does not implement the prompt method!
            def _generate_structured_response(
                self,
                prompt: Union[str, List[Message]],
                response_model: Type[T],
                model_used: str,
            ) -> T:
                return MagicMock(
                    spec=BaseModel
                )  # Dummy for demonstration.  Important for testing!

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

        config = DummyLLMClientConfig(
            can_use_instructor=True,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )
        with self.assertRaises(TypeError):
            BadClient(config=config)  # type: ignore

    def test_abstract_generate_structured_response(self):
        # Test that inheriting classes MUST implement _generate_structured_response method

        class BadClient(SyncLLMClient):
            def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
                return "Test"  # Dummy implementation. Important for testing!

            # Does not implement the _generate_structured_response!

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

            def get_embedding(
                self, prompt_list: List[str], model: Optional[str] = None
            ) -> List[List[float]]:
                # Dummy implementation for testing
                return [[0.0] * 768 for _ in prompt_list]

        config = DummyLLMClientConfig(
            can_use_instructor=True,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )
        with self.assertRaises(TypeError) as context:
            BadClient(config=config)  # type: ignore
            # ignore type by type checker as we are deliberately instantiating with the wrong type

        # Python versions vary in wording; assert on stable substring + method name.
        self.assertIn(
            "Can't instantiate abstract class BadClient", str(context.exception)
        )
        self.assertIn("_generate_structured_response", str(context.exception))

    def test_abstract_get_embedding(self):
        # Test that inheriting classes MUST implement get_embedding method

        class BadClient(SyncLLMClient):
            def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
                return "Test"  # Dummy implementation. Important for testing!

            def _generate_structured_response(
                self,
                prompt: Union[str, List[Message]],
                response_model: Type[T],
                model_used: str,
            ) -> T:
                return MagicMock(
                    spec=BaseModel
                )  # Dummy for demonstration.  Important for testing!

            # Does not implement the get_embedding method!

            @property
            def enum_name(self) -> SyncLLMClientEnum:
                raise NotImplementedError()

        config = DummyLLMClientConfig(
            can_use_instructor=True,
            api_key="dummy_api_key",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=MagicMock(),
        )
        with self.assertRaises(TypeError) as context:
            BadClient(config=config)  # type: ignore
            # ignore type by type checker as we are deliberately instantiating with the wrong type

        # Python versions vary in wording; assert on stable substring + method name.
        self.assertIn(
            "Can't instantiate abstract class BadClient", str(context.exception)
        )
        self.assertIn("get_embedding", str(context.exception))
