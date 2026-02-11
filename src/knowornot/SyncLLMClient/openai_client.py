from typing import List, Optional, Type, TypeVar, Union
import instructor
from openai import OpenAI
from pydantic import BaseModel

from ..config import OpenAIConfig, ToolType, Tool
from ..SyncLLMClient.exceptions import InitialCallFailedException
from ..SyncLLMClient import SyncLLMClient, Message, SyncLLMClientEnum


T = TypeVar("T", bound=BaseModel)


class SyncOpenAIClient(SyncLLMClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)

        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = config.logger
        self.instructor_client = instructor.from_openai(
            self.client, mode=instructor.Mode.TOOLS
        )

        # Check if the tools are compatible with the model
        if self.config.tools:
            for tool in self.config.tools:
                self._check_tool(tool, self.config.default_model)

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )
        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
        )

    def _check_tool(self, tool: Tool, model: str) -> None:
        """
        Checks if a tool requires a specific model configuration.

        Args:
            tool: The tool to check

        Returns:
            bool: True if the tool is compatible with the current model configuration,
                  False if the tool requires a different model
        """
        supported_models = ["gpt-4o-search-preview"]
        if tool.type == ToolType.SEARCH and model not in supported_models:
            raise ValueError(
                f"Model {model} is not supported for search queries for ßOpenAI."
            )

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        """
        Constructs and sends a prompt to the OpenAI chat model and returns the generated response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the OpenAI chat model. It then sends the
        constructed messages to the model specified by `ai_model` and returns the model's textual response.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.
            ai_model: The identifier of the AI model to use for generating the chat response.

        Returns:
            The text content of the first message choice returned by the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """
        messages = self._convert_to_chat_messages(prompt)

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
        """
        Constructs and sends a prompt to the OpenAI chat model and returns the generated structured response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the OpenAI chat model. It then sends the
        constructed messages to the model specified by `model_used` and returns the model's response parsed into an
        instance of `response_model`.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.
            response_model: The type of the response model to parse the output into.
            model_used: The identifier of the AI model to use for generating the chat response.

        Returns:
            An instance of `response_model` containing the structured response from the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """
        messages = self._convert_to_chat_messages(prompt)

        try:
            # For structured responses, we don't want to use tools as instructor handles the response format
            response = self.instructor_client.chat.completions.create(
                model=model_used,
                response_model=response_model,
                messages=messages,
            )
        except Exception as e:
            # Extract the error message directly from the exception
            error_message = str(e)
            # If we get a 404 error about tools not being supported, try without tools mode
            if "404" in error_message:
                self.logger.warning(
                    f"Instructor does not support structured responses for model {model_used}. Falling back to standard completion using custom prompt templates and additional LLM client verification. Note that this increases the cost of the request."
                )

                # Create a new instructor client without tools mode
                json_template_messages = self._apply_json_message_template(
                    prompt, response_model
                )

                response_openai = self.client.chat.completions.create(
                    model=model_used,
                    messages=self._convert_to_chat_messages(json_template_messages),
                )

                if response_openai.choices[0].message.content:
                    raw_output = response_openai.choices[0].message.content.strip()

                    try:
                        rewrite_messages = Message(
                            role="user",
                            content=f"Extract the json object from this string: {raw_output}",
                        )

                        # Use a smaller OpenAI model to rewrite into a structured response
                        response = self.instructor_client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            response_model=response_model,
                            messages=self._convert_to_chat_messages([rewrite_messages]),
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to parse manually structured response {raw_output}. Consider using a different model."
                        ) from e
                else:
                    raise RuntimeError(
                        "Failed to extract response. Consider using a different model."
                    )

            else:
                # Re-raise if it's a different error
                self.logger.error(
                    f"Error generating structured response: {str(e)}. Try a different model."
                )
                raise

        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        if not model:
            model = "text-embedding-3-large"
        embedding_response = self.client.embeddings.create(
            input=prompt_list, model=model
        )
        return list(map(lambda x: x.embedding, embedding_response.data))

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.OPENAI
