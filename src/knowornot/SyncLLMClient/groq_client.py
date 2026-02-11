from typing import List, Optional, Type, TypeVar, Union
import instructor
from groq import Groq
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from groq.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from groq.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from groq.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from ..config import GroqConfig
from .exceptions import InitialCallFailedException
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncGroqClient(SyncLLMClient):
    def __init__(self, config: GroqConfig, base_url: Optional[str] = None):
        super().__init__(config)
        self.config = config
        if base_url:
            self.client = Groq(api_key=config.api_key, base_url=base_url)
        else:
            self.client = Groq(api_key=config.api_key)
        self.logger = config.logger
        self.instructor_client = instructor.from_groq(
            self.client, mode=instructor.Mode.TOOLS
        )

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )
        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
        )

    def _convert_to_groq_chat_messages(
        self, prompt: Union[str, List[Message]]
    ) -> List[ChatCompletionMessageParam]:
        messages: List[ChatCompletionMessageParam] = []

        if isinstance(prompt, str):
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": prompt,
            }
            messages.append(user_message)
        else:
            for m in prompt:
                if m.role == "user":
                    user_message: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": m.content,
                    }
                    messages.append(user_message)
                elif m.role == "system":
                    system_message: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": m.content,
                    }
                    messages.append(system_message)
                elif m.role == "assistant":
                    assistant_message: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": m.content,
                    }
                    messages.append(assistant_message)

        return messages

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        messages = self._convert_to_groq_chat_messages(prompt)

        response = self.client.chat.completions.create(
            model=ai_model,
            messages=messages,
        )
        output = response.choices[0].message.content
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
        messages = self._convert_to_chat_messages(prompt)
        response = self.instructor_client.chat.completions.create(
            model=model_used,
            response_model=response_model,
            messages=messages,
        )
        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError("Groq does not support embeddings")

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.GROQ
