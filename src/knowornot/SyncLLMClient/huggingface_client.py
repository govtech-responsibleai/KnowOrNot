from typing import Type, TypeVar, Union, List, Optional
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionInputJSONSchema,
    ChatCompletionInputMessage,
    ChatCompletionInputResponseFormatJSONSchema,
)
from pydantic import BaseModel

from ..config import HuggingFaceConfig
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncHuggingFaceClient(SyncLLMClient):
    def __init__(self, config: HuggingFaceConfig):
        super().__init__(config)
        self.config = config
        self.logger = config.logger
        self.client = InferenceClient(
            provider=config.provider,
            api_key=config.api_key,
            bill_to=config.bill_to,
        )

    def _convert_messages(
        self, prompt: Union[str, List[Message]]
    ) -> List[ChatCompletionInputMessage]:
        """
        Converts the input prompt into a list of messages in a format suitable for HuggingFace InferenceClient.

        Args:
            prompt: The input prompt to convert. It can be a string or a list of `Message` objects.

        Returns:
            A list of messages in the format required by HuggingFace InferenceClient.
        """
        messages: List[ChatCompletionInputMessage] = []
        if isinstance(prompt, str):
            messages.append(ChatCompletionInputMessage(role="user", content=prompt))
        else:
            for m in prompt:
                messages.append(
                    ChatCompletionInputMessage(role=m.role, content=m.content)
                )
        return messages

    def _add_strict_prompt(
        self, messages: List[ChatCompletionInputMessage]
    ) -> List[ChatCompletionInputMessage]:
        """
        Modifies the input messages by adding a strict prompt to any user messages,
        which enforces the output format of the model. This is done after experimenting with HuggingFace's
        InferenceClient and finding that response_format is not sufficient in ensuring strict output.

        Args:
            messages (List[Dict]): A list of messages to modify.

        Returns:
            List[Dict]: A list of modified messages.
        """
        strict_messages: List[ChatCompletionInputMessage] = []
        for message in messages:
            if message.role == "user" and isinstance(message.content, str):
                strict_messages.append(
                    ChatCompletionInputMessage(
                        role=message.role,
                        content=message.content
                        + "\nOnly use integer or exactly `no citation` for citation -- nothing else",
                    )
                )
            else:
                strict_messages.append(message)
        return strict_messages

    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        response_format = ChatCompletionInputResponseFormatJSONSchema(
            type="json_schema",
            json_schema=ChatCompletionInputJSONSchema(
                name=response_model.__name__,
                schema=response_model.model_json_schema(),
                strict=True,
            ),
        )

        messages = self._convert_messages(prompt)
        messages_union: list[ChatCompletionInputMessage | dict] = [m for m in messages]
        if response_model.__name__ == "QAResponse":
            messages = self._add_strict_prompt(messages)
            messages_union = [m for m in messages]

        response = self.client.chat_completion(
            messages=messages_union,
            response_format=response_format,
            model=model_used,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Expected non-empty content from Hugging Face response")
        structured_data = response_model.parse_raw(content)

        return structured_data

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        messages = self._convert_messages(prompt)
        messages_union: list[ChatCompletionInputMessage | dict] = [m for m in messages]
        response = self.client.chat_completion(
            messages=messages_union,
            model=ai_model,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Expected non-empty content from Hugging Face response")
        return content

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError("Hugging Face client does not support embeddings")

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.HUGGINGFACE
