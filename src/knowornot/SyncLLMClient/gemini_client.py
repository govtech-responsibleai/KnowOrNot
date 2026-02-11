from typing import List, Optional, Type, TypeVar, Union
import warnings

import instructor
from google import genai
from google.genai.types import Tool, GoogleSearch, GenerateContentConfig
from google.generativeai.generative_models import GenerativeModel
from pydantic import BaseModel

from ..config import GeminiConfig, ToolType
from .exceptions import InitialCallFailedException
from . import Message, SyncLLMClient, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncGeminiClient(SyncLLMClient):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.config: GeminiConfig = config
        self.client = genai.Client(api_key=config.api_key)
        self.logger = config.logger

        # Configure tools if provided
        self._tool_config = None
        if self.config.tools:
            tool_configs = []
            for tool in self.config.tools:
                if tool.type == ToolType.SEARCH:
                    google_search_tool = Tool(google_search=GoogleSearch())

                    tool_configs.append(google_search_tool)
                else:
                    raise ValueError(
                        f"Tool type {tool.type} is not supported for Gemini."
                    )
            if tool_configs:
                self._tool_config = GenerateContentConfig(tools=tool_configs)

        # Initialize instructor client
        with warnings.catch_warnings():
            self.instructor_client = instructor.from_gemini(
                client=GenerativeModel(model_name=self.config.default_model),
                mode=instructor.Mode.GEMINI_JSON,
                use_async=False,
            )

            # We should change this to `from_genai`: https://python.useinstructor.com/integrations/genai/#basic-usage
            # `from_genai` uses the correct Python SDK that is also compatible with Vertex AI
            # However, `from_genai` in Instructor needs to be updated to support `Optional` in pydantic schemas

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )

        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
        )

    def _prompt(
        self,
        prompt: Union[str, List[Message]],
        ai_model: str,
    ) -> str:
        """Send a prompt to the Gemini API and get a response.

        Args:
            prompt: The input prompt to send to the LLM. It can be either a single
                string or a list of Message objects.
            ai_model: The name of the AI model to use for generating text.

        Returns:
            The text content of the response from the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """
        if isinstance(prompt, list):
            # Convert messages to a single string
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in prompt])

        response = self.client.models.generate_content(
            model=ai_model, contents=prompt, config=self._tool_config
        )

        output = response.text
        if not output:
            raise ValueError(
                f"Expected output that was not none for {prompt} but got {output}"
            )

        return output

    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        """Generate a structured response from the Gemini API using instructor.

        Args:
            prompt: The input prompt to send to the LLM.
            response_model: The Pydantic model type that the response should be structured into.
            model_used: The name of the AI model to use.

        Returns:
            An instance of `response_model` containing the structured response from the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """
        if model_used != self.config.default_model:
            raise ValueError(
                "Gemini does not support a different model than the default model. Please instantiate a separate Gemini client for different models."
            )

        messages = self._convert_to_chat_messages(prompt)

        # Use instructor to generate structured response
        response = self.instructor_client.chat.completions.create(
            messages=messages, response_model=response_model, tools=self._tool_config
        )

        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings for a list of prompts using the Gemini API.

        Args:
            prompt_list: List of prompts to get embeddings for.
            model: Optional model to use for embeddings. If not provided, uses the default
                embedding model from the config.

        Returns:
            List of embeddings, where each embedding is a list of floats.
        """
        model_to_use = model or self.config.default_embedding_model
        embeddings = []

        for prompt in prompt_list:
            response = self.client.models.embed_content(
                model=model_to_use,
                contents=prompt,
            )
            embeddings.append(response.embeddings)

        return embeddings

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.GEMINI
