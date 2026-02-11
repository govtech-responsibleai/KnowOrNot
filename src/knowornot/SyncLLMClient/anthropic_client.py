from typing import List, Optional, Type, TypeVar, Union
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

from ..config import AnthropicConfig
from .exceptions import InitialCallFailedException
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncAnthropicClient(SyncLLMClient):
    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.config = config
        self.client = Anthropic(api_key=config.api_key)
        self.logger = config.logger
        self.instructor_client = instructor.from_anthropic(self.client)

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )
        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
        )

    def _convert_messages(self, prompt: Union[str, List[Message]]):
        """
        Converts the input prompt into a list of messages in a format suitable for the Anthropic API.

        Args:
            prompt: The input prompt to convert. It can be a string or a list of `Message` objects.

        Returns:
            A list of messages in the format required by the Anthropic API.
        """
        messages = []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            for m in prompt:
                messages.append({"role": m.role, "content": m.content})
        return messages

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        messages = self._convert_messages(prompt)
        response = self.client.messages.create(
            model=ai_model,
            messages=messages,
            max_tokens=1024,
        )
        output = None
        if response.content:
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    output = text
                    break
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
        messages = self._convert_messages(prompt)
        response = self.instructor_client.messages.create(
            model=model_used,
            messages=messages,
            response_model=response_model,
            max_tokens=1024,
        )
        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError("Anthropic client does not support embeddings")

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.ANTHROPIC
