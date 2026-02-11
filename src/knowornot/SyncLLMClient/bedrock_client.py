from typing import (
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    Literal,
    TypedDict,
    cast,
)
import instructor.client_bedrock as instructor_bedrock
import boto3
from botocore.client import BaseClient
from pydantic import BaseModel

from ..config import BedrockConfig
from .exceptions import InitialCallFailedException
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class BedrockTextBlock(TypedDict):
    text: str


class BedrockMessage(TypedDict):
    role: Literal["user"]
    content: List[BedrockTextBlock]


class BedrockSystemBlock(TypedDict):
    text: str


class BedrockInstructorClient(Protocol):
    def create(self, **kwargs: object) -> BaseModel: ...


class BedrockInstructorWrapper:
    def __init__(self, instructor_client: BedrockInstructorClient):
        self._client = instructor_client

    def create(
        self,
        *,
        response_model: Type[T],
        modelId: str,
        messages: List[BedrockMessage],
        system: Optional[List[BedrockSystemBlock]],
    ) -> T:
        result = self._client.create(
            response_model=response_model,
            modelId=modelId,
            messages=messages,
            system=system,
        )
        return cast(T, result)


def _bedrock_converse_sync(
    client: BaseClient,
    model_name: str,
    user_prompt: str,
    system_prompt: str | None = None,
) -> str:
    messages: List[BedrockMessage] = [
        {"role": "user", "content": [{"text": user_prompt}]}
    ]
    kwargs = {"modelId": model_name, "messages": messages}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]
    response = client.converse(**kwargs)
    return response["output"]["message"]["content"][0]["text"]


class SyncBedrockClient(SyncLLMClient):
    def __init__(self, config: BedrockConfig):
        super().__init__(config)
        self.config = config
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=config.region_name,
        )
        self.logger = config.logger
        self.instructor_client = BedrockInstructorWrapper(
            cast(BedrockInstructorClient, instructor_bedrock.from_bedrock(self.client))
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

    def _convert_messages(self, prompt: Union[str, List[Message]]):
        """
        Converts the input prompt, which can be a string or a list of `Message` objects,
        into a user prompt and an optional system prompt suitable for the Bedrock
        converse API.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.

        Returns:
            A tuple of the user prompt as a string and the system prompt as an optional string.
        """
        user_prompt = ""
        system_prompt = None
        if isinstance(prompt, str):
            user_prompt = prompt
        else:
            parts = []
            for m in prompt:
                if m.role == "system":
                    system_prompt = m.content
                elif m.role == "user":
                    parts.append(m.content)
            user_prompt = "\n".join(parts)
        return user_prompt, system_prompt

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        user_prompt, system_prompt = self._convert_messages(prompt)
        return _bedrock_converse_sync(self.client, ai_model, user_prompt, system_prompt)

    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        user_prompt, system_prompt = self._convert_messages(prompt)
        response = self.instructor_client.create(
            modelId=model_used,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": system_prompt}] if system_prompt else None,
            response_model=response_model,
        )
        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError("Bedrock client does not support embeddings")

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.BEDROCK
