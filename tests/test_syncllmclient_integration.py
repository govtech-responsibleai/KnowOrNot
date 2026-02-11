# tests/test_syncllmclient_integration.py

import os
import unittest

from dotenv import load_dotenv

from src.knowornot import KnowOrNot
from src.knowornot.SyncLLMClient import SyncLLMClientEnum
from src.knowornot.common.models import QAResponse

load_dotenv()


class TestSyncLLMClientIntegration(unittest.TestCase):
    def _require_env(self, *keys: str) -> None:
        if os.environ.get("RUN_LLM_INTEGRATION_TESTS") != "1":
            self.skipTest("Set RUN_LLM_INTEGRATION_TESTS=1 to run integration tests")
        missing = [key for key in keys if not os.environ.get(key)]
        if missing:
            self.skipTest(f"Missing env vars: {', '.join(missing)}")

    def test_openai_client(self):
        self._require_env(
            "OPENAI_API_KEY",
            "OPENAI_DEFAULT_MODEL",
            "OPENAI_DEFAULT_EMBEDDING_MODEL",
        )
        kon = KnowOrNot()
        kon.add_openai()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.OPENAI))

    def test_azure_openai_client(self):
        self._require_env(
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEFAULT_MODEL",
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL",
        )
        kon = KnowOrNot()
        kon.add_azure()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.AZURE_OPENAI))

    def test_anthropic_client(self):
        self._require_env("ANTHROPIC_API_KEY", "ANTHROPIC_DEFAULT_MODEL")
        kon = KnowOrNot()
        kon.add_anthropic()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.ANTHROPIC))

    def test_bedrock_client(self):
        self._require_env("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_DEFAULT_MODEL")
        kon = KnowOrNot()
        kon.add_bedrock()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.BEDROCK))

    def test_gemini_client(self):
        self._require_env(
            "GEMINI_API_KEY",
            "GEMINI_DEFAULT_MODEL",
            "GEMINI_DEFAULT_EMBEDDING_MODEL",
        )
        kon = KnowOrNot()
        kon.add_gemini()
        self._test_client_string(kon.get_client(SyncLLMClientEnum.GEMINI))

    def test_openrouter_client(self):
        self._require_env("OPENROUTER_API_KEY", "OPENROUTER_DEFAULT_MODEL")
        kon = KnowOrNot()
        kon.add_openrouter()
        self._test_client_string(kon.get_client(SyncLLMClientEnum.OPENROUTER))

    def test_huggingface_client(self):
        self._require_env(
            "HUGGINGFACE_API_KEY",
            "HUGGINGFACE_PROVIDER",
            "HUGGINGFACE_DEFAULT_MODEL",
        )
        kon = KnowOrNot()
        kon.add_huggingface()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.HUGGINGFACE))

    def _test_client_structured(self, client):
        prompt = "What is the capital of France?"
        response = client.get_structured_response(prompt, QAResponse)
        self.assertIsInstance(response, QAResponse)
        self.assertIsNotNone(response.response)

    def _test_client_string(self, client):
        prompt = "What is the capital of France?"
        response = client.prompt(prompt)
        self.assertIsInstance(response, str)
