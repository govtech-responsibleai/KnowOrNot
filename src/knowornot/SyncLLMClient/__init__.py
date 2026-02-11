from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Type, TypeVar, Union, overload, Literal
import re

from pydantic import BaseModel

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from ..config import LLMClientConfig

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=Enum)


class Message(BaseModel):
    role: str
    content: str


class SyncLLMClientEnum(Enum):
    AZURE_OPENAI = "AZURE_OPENAI"
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    OPENROUTER = "OPENROUTER"
    GROQ = "GROQ"
    ANTHROPIC = "ANTHROPIC"
    BEDROCK = "BEDROCK"
    HUGGINGFACE = "HUGGINGFACE"


class SyncLLMClient(ABC):
    def __init__(self, config: LLMClientConfig):
        self.config = config
        self.logger = self.config.logger

    @abstractmethod
    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        pass

    def _apply_json_message_template(
        self, prompt: Union[str, List[Message]], json_model: Type[T]
    ) -> List[Message]:
        """
        Applies a template to user and system messages in the list.
        This is used to generate a structured response from the LLM, if Instructor does not support the model provided.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.
            json_model: The Pydantic model type that the response should be structured into

        Returns:
            List[Message]: New list of messages with template applied
        """

        system_prompt = "You are a helpful assistant. Always reply ONLY in JSON, following the exact format given."

        def build_user_prompt(query: str, json_model: Type[T]) -> str:
            # Create a simple example structure
            return f"""
                Please provide the information using this Pydantic model schema:

                {json_model.model_json_schema()}

                Prompt:
                {query}

                Only return valid JSON. Do not include any explanation or additional text.
                IMPORTANT: All strings in the JSON must be properly escaped. Use \\n for newlines and escape any quotes with \\
                """

        if isinstance(prompt, str):
            return [
                Message(role="system", content=system_prompt),
                Message(role="user", content=build_user_prompt(prompt, json_model)),
            ]

        messages: List[Message] = []

        for m in prompt:
            if m.role == "user":
                user_message = Message(
                    role="user",
                    content=build_user_prompt(m.content, json_model),
                )
                messages.append(user_message)
            elif m.role == "system":
                system_message = Message(
                    role="system", content=system_prompt + m.content
                )
                messages.append(system_message)
            elif m.role == "assistant":
                assistant_message = Message(role="assistant", content=m.content)
                messages.append(assistant_message)

        return messages

    def _convert_to_chat_messages(
        self, prompt: Union[str, List[Message]]
    ) -> List[ChatCompletionMessageParam]:
        """
        Converts a prompt (either string or list of Message objects) into a list of ChatCompletionMessageParam.

        Args:
            prompt: Either a string or a list of Message objects

        Returns:
            List[ChatCompletionMessageParam]: List of messages in OpenAI chat format
        """
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

    def prompt(
        self, prompt: Union[str, List[Message]], ai_model: Optional[str] = None
    ) -> str:
        """Sends a prompt to the LLM and gets a text response.

        Args:
            prompt: The input prompt to send to the LLM. It can be either a single
                string or a list of Message objects.
            ai_model: The name of the AI model to use for generating text. If not
                specified, uses the default model configured in the LLMClientConfig.

        Returns:
            The text response from the LLM.
        """
        model_to_use: str = self.config.default_model
        if ai_model is not None:
            model_to_use = ai_model

        self.logger.debug(f"Using model: {model_to_use} and sending prompt {prompt}")

        return self._prompt(prompt=prompt, ai_model=model_to_use)

    @property
    def can_use_instructor(self) -> bool:
        return self.config.can_use_instructor

    @abstractmethod
    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        pass

    def get_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        ai_model: Optional[str] = None,
    ) -> T:
        """
        Generate a structured response from a given prompt using the LLM.

        Parameters:
        - prompt (Union[str, List[Message]]): The input prompt to send to the LLM. It can be
        either a single string or a list of Message objects.
        - response_model (Type[T]): The Pydantic model type that the response should be
        structured into.

        Returns:
        - T: An instance of the response model containing the structured response.

        Raises:
        - ValueError: If the instructor mode is not enabled in the configuration, which is
        required to generate structured responses.
        """

        if not self.can_use_instructor:
            raise ValueError(
                "This LLM client cannot generate structured responses. "
                "Enable instructor mode in the configuration to use this feature."
            )

        model_to_use: str = ai_model or self.config.default_model

        self.logger.debug(f"Using model: {model_to_use} and sending prompt {prompt}")

        return self._generate_structured_response(
            prompt=prompt, response_model=response_model, model_used=model_to_use
        )

    @abstractmethod
    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        pass

    @overload
    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["error", "first", "last"],
        ai_model: Optional[str] = None,
    ) -> str: ...

    @overload
    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["list"],
        ai_model: Optional[str] = None,
    ) -> List[str]: ...

    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["error", "first", "last", "list"],
        ai_model: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Prompts the LLM and extracts values from XML tags in the response, validating against an allowed list.

        Args:
            prompt: The input prompt to send to the LLM. Can be a string or list of Message objects.
            tag_name: The name of the XML tag to extract from.
            allowed_list: The list of allowed values to validate the extracted value(s) against.
            ai_model: The name of the AI model to use. If None, uses the default model.
            on_multiple: How to handle multiple tag matches. Options:
                - "error": Raise ValueError if multiple tags are found (default)
                - "first": Return only the first match
                - "last": Return only the last match
                - "list": Return all matches as a list

        Returns:
            If on_multiple is "first", "last", or "error" (and only one match): The extracted value
            If on_multiple is "list": A list of extracted values

        Raises:
            ValueError: If no tags are found, if the extracted value isn't in the allowed list,
                       or if multiple tags are found when on_multiple="error"
        """
        # Get response from LLM
        response = self.prompt(prompt=prompt, ai_model=ai_model)

        # Create regex pattern for the specified tag
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            raise ValueError(f"No <{tag_name}> tags found in LLM response")

        # Handle multiple matches according to on_multiple parameter
        if len(matches) > 1:
            if on_multiple == "error":
                raise ValueError(f"Multiple <{tag_name}> tags found in LLM response")
            elif on_multiple == "first":
                matches = [matches[0]]
            elif on_multiple == "last":
                matches = [matches[-1]]
            elif on_multiple != "list":
                raise ValueError(f"Invalid on_multiple value: {on_multiple}")

        # Validate against allowed list and convert
        result = []
        for value in matches:
            value = value.strip()
            if value not in allowed_list:
                raise ValueError(
                    f"Value '{value}' from tag <{tag_name}> is not in the allowed list: {allowed_list}"
                )
            result.append(value)

        # Return result based on on_multiple
        return result if on_multiple == "list" else result[0]

    def prompt_and_extract_tags(
        self,
        prompt: Union[str, List[Message]],
        validated_tags: Dict[str, List[str]],
        free_tags: Optional[List[str]] = None,
        ai_model: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Prompts the LLM and extracts values from XML tags, with option for validation against allowed lists.

        Args:
            prompt: The input prompt to send to the LLM. Can be a string or list of Message objects.
            validated_tags: Dictionary mapping tag names to their allowed values. These tags must have
                           values in the specified list.
            free_tags: List of tag names that can contain any content (no validation).
            ai_model: The name of the AI model to use. If None, uses the default model.

        Returns:
            Dictionary mapping tag names to lists of their extracted values.

        Raises:
            ValueError: If validation fails, or if any required tags are missing from the response.
        """
        if free_tags is None:
            free_tags = []

        for tag_name, allowed_values in validated_tags.items():
            if not isinstance(allowed_values, list):
                raise ValueError(f"Allowed values for {tag_name} must be a list")
            if not allowed_values:
                raise ValueError(f"Allowed values list for {tag_name} cannot be empty")

        all_tags = list(validated_tags.keys()) + free_tags

        if not all_tags:
            raise ValueError("At least one tag (validated or free) must be specified")

        response = self.prompt(prompt=prompt, ai_model=ai_model)

        results = {}

        for tag_name in all_tags:
            pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
            matches: List[str] = re.findall(pattern, response, re.DOTALL)

            if not matches:
                raise ValueError(f"No <{tag_name}> tags found in LLM response")

            if tag_name in validated_tags:
                allowed_values = validated_tags[tag_name]
                validated_values = []

                for value in matches:
                    value = value.strip()
                    if value not in allowed_values:
                        raise ValueError(
                            f"Value '{value}' from tag <{tag_name}> is not in the allowed list: {allowed_values}"
                        )
                    validated_values.append(value)

                results[tag_name] = validated_values
            else:
                results[tag_name] = [match.strip() for match in matches]

        return results

    @property
    @abstractmethod
    def enum_name(self) -> SyncLLMClientEnum:
        raise NotImplementedError()
