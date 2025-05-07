import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
import logging

from .SyncLLMClient.openai_client import SyncOpenAIClient

from .QuestionExtractor.models import FilterMethod
from .common.models import (
    EvaluatedExperimentDocument,
    ExperimentInputDocument,
    ExperimentOutputDocument,
    LabeledDataSample,
    Prompt,
    QAPair,
    QAPairFinal,
    QuestionDocument,
    RetrievalType,
    EvaluationSpec,
)
from .QuestionExtractor import QuestionExtractor
from .config import AzureOpenAIConfig, OpenAIConfig
from .SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
from .SyncLLMClient.azure_client import SyncAzureOpenAIClient
from .FactManager import FactManager
from .PromptManager import PromptManager
from .ExperimentManager.models import ExperimentParams, ExperimentType
from .ExperimentManager import ExperimentManager
from .Evaluator import Evaluator
from .DataLabeller import DataLabeller

__all__ = ["KnowOrNot", "SyncLLMClient"]


class KnowOrNot:
    def __init__(self):
        """
        Initialize the KnowOrNot instance with the given configuration.
        """
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.client_registry: Dict[SyncLLMClientEnum, SyncLLMClient] = {}
        self.default_sync_client: Optional[SyncLLMClient] = None
        self.fact_manager: Optional[FactManager] = None
        self.question_manager: Optional[QuestionExtractor] = None
        self.experiment_manager: Optional[ExperimentManager] = None
        self.evaluator: Optional[Evaluator] = None
        self.data_labeller: Optional[DataLabeller] = None
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(levelname)-8s %(asctime)s.%(msecs)03d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Config initialized")

    def register_client(
        self, client: SyncLLMClient, make_default: bool = False
    ) -> None:
        """
        Register a new LLM client in the registry.

        Parameters:
        - client (SyncLLMClient): The client to register
        - make_default (bool): Whether to set this as the default client
        """
        enum_name = client.enum_name
        self.client_registry[enum_name] = client

        # Set as default if requested or if it's the first client
        if make_default or self.default_sync_client is None:
            self.default_sync_client = client

    def get_client(
        self, client_enum: Optional[SyncLLMClientEnum] = None
    ) -> SyncLLMClient:
        """
        Get a client by its enum type, or return the default if no type specified.

        Parameters:
        - client_enum (Optional[SyncLLMClientEnum]): Enum type of client to retrieve

        Returns:
        - SyncLLMClient: The requested client instance

        Raises:
        - KeyError: If no client of the specified enum type is registered
        - ValueError: If no default client is available (when client_enum is None)
        """
        if client_enum is None:
            if self.default_sync_client is None:
                raise ValueError("No default client available")
            return self.default_sync_client

        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        return self.client_registry[client_enum]

    def get_all_clients(self) -> List[SyncLLMClient]:
        """
        Get all registered clients.

        Returns:
        - List[SyncLLMClient]: List of all registered client instances
        """
        return list(self.client_registry.values())

    def set_default_client(self, client_enum: SyncLLMClientEnum) -> None:
        """
        Set a specific client as the default.

        Parameters:
        - client_enum (SyncLLMClientEnum): Enum of client to set as default

        Raises:
        - KeyError: If no client of the specified type is registered
        """
        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        self.default_sync_client = self.client_registry[client_enum]

    def remove_client(self, client_enum: SyncLLMClientEnum) -> None:
        """
        Remove a client from the registry.

        Parameters:
        - client_enum (SyncLLMClientEnum): Enum of client to remove

        Raises:
        - KeyError: If no client of the specified type is registered
        - ValueError: If attempting to remove the default client
        """
        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        if self.default_sync_client is self.client_registry[client_enum]:
            raise ValueError(
                "Cannot remove the default client. Set a new default first."
            )

        del self.client_registry[client_enum]

    def _get_fact_manager(
        self, alternative_client: Optional[SyncLLMClient] = None
    ) -> FactManager:
        if self.fact_manager is not None:
            return self.fact_manager

        client = alternative_client or self.default_sync_client

        if not client:
            raise ValueError(
                "default client is not set and no alternative client has been provided"
            )

        self.fact_manager = FactManager(
            sync_llm_client=client,
            default_fact_creation_prompt=PromptManager.default_fact_extraction_prompt,
            logger=self.logger,
        )

        return self.fact_manager

    def _get_question_manager(
        self, alternative_llm_client: Optional[SyncLLMClient]
    ) -> QuestionExtractor:
        if self.question_manager is not None:
            return self.question_manager

        client = alternative_llm_client or self.default_sync_client

        if not client:
            raise ValueError(
                "You must set a default LLM Client or pass one in before performing any question related operations"
            )

        self.question_manager = QuestionExtractor(
            question_prompt_default=PromptManager.default_question_extraction_prompt,
            default_client=client,
            logger=self.logger,
        )

        return self.question_manager

    def _get_experiment_manager(
        self,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        alternative_hypothetical_answer_prompt: Optional[str] = None,
    ) -> ExperimentManager:
        client = alternative_llm_client or self.default_sync_client
        if not client:
            raise ValueError(
                "You must set a default LLM Client or pass one in before performing any experiment related operations"
            )

        prompt = (
            alternative_hypothetical_answer_prompt
            or PromptManager.hypothetical_answer_generator
        )

        self.experiment_manager = ExperimentManager(
            default_client=client, logger=self.logger, hypothetical_answer_prompt=prompt
        )

        return self.experiment_manager

    def _get_evaluator(
        self,
        evaluation_spec_dict: Dict[str, EvaluationSpec],
        alternative_llm_client: Optional[SyncLLMClient] = None,
        model_to_use: Optional[str] = None,
    ) -> Evaluator:
        client = alternative_llm_client or self.default_sync_client
        if not client:
            raise ValueError(
                "You must set a default LLM Client or pass one in before performing any experiment related operations"
            )

        self.evaluator = Evaluator(
            default_client=client,
            logger=self.logger,
            evaluation_dict=evaluation_spec_dict,
            evaluation_model=model_to_use,
        )

        return self.evaluator

    def _get_data_labeller(
        self,
        logger: logging.Logger,
    ) -> DataLabeller:
        if self.data_labeller is not None:
            return self.data_labeller

        self.data_labeller = DataLabeller(
            logger=logger,
        )

        return self.data_labeller

    def add_azure(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
    ) -> None:
        """
        Create a `KnowOrNot` instance using Azure configuration details.

        This is the recommended way to instantiate a KnowOrNot instance.

        Parameters:
        - azure_endpoint (Optional[str]): The Azure endpoint. If not provided,
        the environment variable `AZURE_OPENAI_ENDPOINT` will be used.
        - azure_api_key (Optional[str]): The Azure API key. If not provided,
        the environment variable `AZURE_OPENAI_API_KEY` will be used.
        - azure_api_version (Optional[str]): The Azure API version. If not provided,
        the environment variable `AZURE_OPENAI_API_VERSION` will be used.
        - default_model (Optional[str]): The default model for synchronous operations.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_MODEL` will be used.
        - default_embedding_model (Optional[str]): The default embedding model.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL` will be used.

        Returns:
        - KnowOrNot: An instance of the `KnowOrNot` class configured with the specified
        Azure settings.

        Example:
        1. Using environment variables: KnowOrNot.create_from_azure()

        2. Providing all parameters: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_api_version="2023-05-15",
                default_model="gpt-4o",
                default_embedding_model="text-embedding-3-large"
            )
        """

        if not azure_endpoint:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise EnvironmentError(
                    "AZURE_OPENAI_ENDPOINT is not set and azure_endpoint is not provided"
                )
        if not azure_api_key:
            azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not azure_api_key:
                raise EnvironmentError(
                    "AZURE_OPENAI_API_KEY is not set and azure_api_key is not provided"
                )
        if not azure_api_version:
            azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
            if not azure_api_version:
                raise EnvironmentError(
                    "AZURE_OPENAI_API_VERSION is not set and azure_api_version is not provided"
                )
        if not default_model:
            default_model = os.environ.get("AZURE_OPENAI_DEFAULT_MODEL")
            if not default_model:
                raise EnvironmentError(
                    "AZURE_OPENAI_DEFAULT_MODEL is not set and default_model is not provided"
                )
        if not default_embedding_model:
            default_embedding_model = os.environ.get(
                "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL"
            )
            if not default_embedding_model:
                raise EnvironmentError(
                    "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL is not set and default_embedding_model is not provided"
                )

        logger = logging.getLogger(__name__)

        azure_config = AzureOpenAIConfig(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
            default_model=default_model,
            default_embedding_model=default_embedding_model,
            logger=logger,
        )

        azure_sync_client = SyncAzureOpenAIClient(config=azure_config)

        self.register_client(client=azure_sync_client, make_default=True)

        return

    def add_openai(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        project: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> None:
        """
        Registers an OpenAI API client with the KnowOrNot instance.

        If ``api_key`` is not provided, the value of the ``OPENAI_API_KEY`` environment variable is used.
        If ``default_model`` is not provided, the value of the ``OPENAI_DEFAULT_MODEL`` environment variable is used.
        If ``default_embedding_model`` is not provided, the value of the ``OPENAI_DEFAULT_EMBEDDING_MODEL`` environment variable is used.

        Args:
            api_key (str, optional): The API key to use. Must be provided or available in the environment.
            default_model (str, optional): The model to use by default. Must be provided or available in the environment.
            default_embedding_model (str, optional): The embedding model to use by default. Must be provided or available in the environment.
            project (str, optional): The project to associate with the client. Defaults to ``None``.
            organization (str, optional): The organization to associate with the client. Defaults to ``None``.

        Raises:
            EnvironmentError: If ``api_key``, ``default_model``, or ``default_embedding_model`` are not provided and not found in the environment.

        Returns:
            None
        """
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set and api_key is not provided"
                )
        if not default_model:
            default_model = os.environ.get("OPENAI_DEFAULT_MODEL")
            if not default_model:
                raise EnvironmentError(
                    "OPENAI_DEFAULT_MODEL is not set and default_model is not provided"
                )
        if not default_embedding_model:
            default_embedding_model = os.environ.get("OPENAI_DEFAULT_EMBEDDING_MODEL")
            if not default_embedding_model:
                raise EnvironmentError(
                    "OPENAI_DEFAULT_EMBEDDING_MODEL is not set and default_embedding_model is not provided"
                )

        config = OpenAIConfig(
            logger=self.logger,
            api_key=api_key,
            default_model=default_model,
            default_embedding_model=default_embedding_model,
            project=project,
            organization=organization,
        )

        openai_sync_client = SyncOpenAIClient(config=config)

        self.register_client(client=openai_sync_client, make_default=True)

        return None

    def create_questions(
        self,
        source_paths: List[Path],
        knowledge_base_identifier: str,
        context_prompt: str,
        path_to_save_questions: Path,
        filter_method: Literal["keyword", "semantic", "both"],
        alternative_fact_creation_llm_prompt: Optional[Prompt] = None,
        alternative_fact_creator_llm_client: Optional[SyncLLMClient] = None,
        alternative_fact_creation_llm_model: Optional[str] = None,
        alternative_question_creation_llm_prompt: Optional[Prompt] = None,
        alternative_question_creator_llm_client: Optional[SyncLLMClient] = None,
        fact_storage_dir: Optional[Path] = None,
        semantic_filter_threshold: Optional[float] = None,
        keyword_filter_threshold: Optional[float] = None,
        intermediate_storage_path: Optional[Path] = None,
    ) -> QuestionDocument:
        """
        Creates diverse question/answer pairs for the documents given.

        Args:
            source_paths(List[Path]): A list of txt files to make the questions from
            knowledge_base_identifier (str): The identifier for the knowledge base
            context_prompt (str): The context to include in the prompt sent to the LLM to generate the questions.
            path_to_save_questions (Path): The path to save the questions to. Must be a json.
            filter_method (Literal["keyword", "semantic", "both"]): The method to use for filtering the facts.
            alternative_fact_creation_llm_prompt (Optional[Prompt]): An alternative prompt to use for fact creation. If not provided, the default prompt from PromptManager will be used.
            alternative_fact_creator_llm_client (Optional[SyncLLMClient]): An alternative client to use for fact creation. If not provided, the default client will be used.
            alternative_fact_creation_llm_model (Optional[str]): An alternative model to use for fact creation. If not provided, the default model of either the client provided or the default client will be used with preference for the client provided.
            alternative_question_creation_llm_prompt (Optional[Prompt]): An alternative prompt to use for question creation. If not provided, the default prompt from PromptManager will be used.
            alternative_question_creator_llm_client (Optional[SyncLLMClient]): An alternative client to use for question creation. If not provided, the default client will be used.
            fact_storage_dir (Optional[Path]): The directory to store the facts in. If not provided, facts will not be stored.
            semantic_filter_threshold (Optional[float]): The threshold to use for semantic filtering. If not provided, the default threshold of 0.3 will be used.
            keyword_filter_threshold (Optional[float]): The threshold to use for keyword filtering. If not provided, the default threshold of 0.3 will be used.
            intermediate_storage_path (Optional[Path]): The json path to store intermediate outputs in. If not provided, intermediate outputs will not be stored.

        Returns:
            QuestionDocument: A QuestionDocument object containing the questions and answers

        Raises:
            TypeError: If any of the provided source paths are not of type Path
            FileNotFoundError: If any of the provided source paths do not exist
            ValueError: If no client is set and no alternative client is provided
            ValueError: If any of the provided source paths are not .txt files
            ValueError: If the provided fact_storage_dir is not a directory
            ValueError: If the provided filter_method is not one of ['keyword', 'semantic', 'both']
            ValueError: If the alternative client provided, or default client cannot use instructor
            ValueError: If the question process is unable to create any diverse questions
            ValueError: If the intermediate_storage_path is provided but is not a .json file

        """

        for path in source_paths:
            if not isinstance(path, Path):
                raise TypeError(f"{path} is not of type Path")
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist")
            if not path.suffix.lower() == ".txt":
                raise ValueError(f"Only .txt files are provided. Got {path}")

        fact_manager = self._get_fact_manager(
            alternative_client=alternative_fact_creator_llm_client
        )

        if fact_storage_dir is not None:
            if not fact_storage_dir.is_dir():
                raise ValueError(
                    f"Expectd fact_storage_dir to be a directory. Got {fact_storage_dir}"
                )

        if filter_method not in ["keyword", "semantic", "both"]:
            raise ValueError(
                f"Expected filter_method to be one of ['keyword', 'semantic', 'both']. Got {filter_method}"
            )

        if (
            intermediate_storage_path
            and not intermediate_storage_path.suffix.lower() == ".json"
        ):
            raise ValueError(
                f"Expected intermediate_storage_path to be .json file. Got {intermediate_storage_path}"
            )

        fact_document_list = fact_manager._parse_sources_to_atomic_facts(
            source_list=source_paths,
            destination_dir=fact_storage_dir,
            alternative_prompt=alternative_fact_creation_llm_prompt.content
            if alternative_fact_creation_llm_prompt
            else None,
            alternative_ai_model=alternative_fact_creation_llm_model,
        )

        question_extractor = self._get_question_manager(
            alternative_llm_client=alternative_question_creator_llm_client
        )

        if filter_method == "keyword":
            filter_method_enum = FilterMethod.KEYWORD
        elif filter_method == "semantic":
            filter_method_enum = FilterMethod.SEMANTIC
        elif filter_method == "both":
            filter_method_enum = FilterMethod.BOTH

        question_document = question_extractor.generate_questions_from_documents(
            knowledge_base_identifier=knowledge_base_identifier,
            documents=fact_document_list,
            context_prompt=context_prompt,
            method=filter_method_enum,
            path_to_save=path_to_save_questions,
            alternative_question_prompt=alternative_question_creation_llm_prompt.content
            if alternative_question_creation_llm_prompt
            else None,
            diversity_threshold_keyword=keyword_filter_threshold,
            diversity_threshold_semantic=semantic_filter_threshold,
            intermediate_storage_path=intermediate_storage_path,
        )

        question_document.save_to_json()

        self.logger.info(
            f"Created question document with {len(question_document.questions)} questions and saved them to {question_document.path_to_store}"
        )

        return question_document

    def create_diverse_questions_from_QAPairs(
        self,
        knowledge_base_identifier: str,
        qa_pairs: List[Dict[str, str]],
        method: Literal["keyword", "semantic", "both"],
        path_to_save: Path,
        diversity_threshold_keyword: float = 0.3,
        diversity_threshold_semantic: float = 0.3,
    ) -> QuestionDocument:
        """
        Filters and creates diverse question-answer pairs from provided QAPairs.

        This function processes a list of question-answer pairs (`qa_pairs`), applies a specified filtering method
        to ensure diversity, and saves the resulting diverse set of questions to a specified path.

        Args:
            knowledge_base_identifier (str): A unique identifier for the knowledge base associated with the questions.
            qa_pairs (List[Dict[str, str]]): A list of dictionaries, each containing a 'question' and 'answer' key.
            method (Literal["keyword", "semantic", "both"]): The filtering method to apply - 'keyword', 'semantic', or 'both'.
            path_to_save (Path): The file path where the filtered questions will be saved.
            diversity_threshold_keyword (float, optional): The threshold for keyword diversity filtering. Defaults to 0.3.
            diversity_threshold_semantic (float, optional): The threshold for semantic diversity filtering. Defaults to 0.3.

        Returns:
            QuestionDocument: A `QuestionDocument` containing the filtered and diverse question-answer pairs.

        Raises:
            ValueError: If the `method` is not one of ['keyword', 'semantic', 'both'].
            ValueError: If any `qa_pair` does not contain 'question' and 'answer' keys.
        """

        if method not in ["keyword", "semantic", "both"]:
            raise ValueError(
                f"Expected method to be one of ['keyword', 'semantic', 'both']. Got {method}"
            )

        _method: FilterMethod
        if method == "keyword":
            _method = FilterMethod.KEYWORD
        elif method == "semantic":
            _method = FilterMethod.SEMANTIC
        elif method == "both":
            _method = FilterMethod.BOTH

        question_manager = self._get_question_manager(alternative_llm_client=None)

        for qa_pair in qa_pairs:
            if "question" not in qa_pair or "answer" not in qa_pair:
                raise ValueError(
                    f"Expected qa_pair to have 'question' and 'answer' keys. Got {qa_pair}"
                )
        qa_pairs_to_use = [QAPair(**qa_pair) for qa_pair in qa_pairs]
        return question_manager.filter_questions(
            identifier=knowledge_base_identifier,
            questions=qa_pairs_to_use,
            method=_method,
            path_to_save=path_to_save,
            diversity_threshold_keyword=diversity_threshold_keyword,
            diversity_threshold_semantic=diversity_threshold_semantic,
        )

    def _make_questions_for_experiment(self, qa_final: QAPairFinal) -> QAPair:
        return QAPair(question=qa_final.question, answer=qa_final.answer)

    def create_experiment_input(
        self,
        question_document: Union[Path, QuestionDocument],
        system_prompt: Prompt,
        experiment_type: Literal["removal", "synthetic"],
        retrieval_type: Literal["DIRECT", "BASIC_RAG", "LONG_IN_CONTEXT", "HYDE_RAG"],
        input_store_path: Path,
        output_store_path: Path,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model_to_use: Optional[str] = None,
        alternative_prompt_for_hyde: Optional[Prompt] = None,
        alternative_llm_client_for_hyde: Optional[SyncLLMClient] = None,
        ai_model_for_hyde: Optional[str] = None,
    ) -> ExperimentInputDocument:
        """Creates an experiment input document based on the provided parameters.

        This function generates an `ExperimentInputDocument` by orchestrating the
        necessary components, including loading the question document,
        validating input parameters, and configuring the experiment manager.
        It supports different experiment types (removal, synthetic) and retrieval
        strategies (DIRECT, BASIC_RAG, LONG_IN_CONTEXT, HYDE_RAG).

        Args:
            question_document (Union[Path, QuestionDocument]): The question document
                to use for the experiment. It can be a Path to a JSON file or an
                already loaded QuestionDocument object.
            system_prompt (Prompt): The system prompt to be used for the experiment.
                This prompt sets the context for the LLM during experiment execution.
            experiment_type (Literal["removal", "synthetic"]): The type of
                experiment to create. "removal" removes questions from the context,
                while "synthetic" uses generated questions.
            retrieval_type (Literal["DIRECT", "BASIC_RAG", "LONG_IN_CONTEXT", "HYDE_RAG"]):
                The retrieval strategy to use for the experiment. Determines how
                context is retrieved for the questions.
            input_store_path (Path): The path to store the generated
                ExperimentInputDocument as a JSON file.
            output_store_path (Path): The path to store the experiment output
                document.
            alternative_llm_client (Optional[SyncLLMClient]): An optional alternative
                LLM client to use for the core experiment. If not provided, the
                default LLM client will be used.
            ai_model_to_use (Optional[str]): An optional AI model to use for the core
                experiment. If not provided, the default model from the chosen LLM
                client will be used.
            alternative_prompt_for_hyde (Optional[Prompt]): An optional alternative
                prompt for the HYDE_RAG retrieval strategy.
            alternative_llm_client_for_hyde (Optional[SyncLLMClient]): An optional
                alternative LLM client to use for the HYDE_RAG retrieval strategy.
            ai_model_for_hyde (Optional[str]): An optional AI model to use for the
                HYDE_RAG retrieval strategy.

        Returns:
            QuestionDocument: The generated ExperimentInputDocument.

        Raises:
            ValueError: If the experiment_type or retrieval_type is invalid.
            ValueError: If no default LLM client is set and no alternative is provided.
        """

        if isinstance(question_document, Path):
            question_document = QuestionDocument.load_from_json(question_document)

        if experiment_type not in ["removal", "synthetic"]:
            raise ValueError(
                f'Expected experiment type to be one of "removal", "synthetic" but got {experiment_type}'
            )

        if experiment_type == "removal":
            experiment_type_enum = ExperimentType.REMOVAL
        elif experiment_type == "synthetic":
            experiment_type_enum = ExperimentType.SYNTHETIC

        if retrieval_type not in ["DIRECT", "BASIC_RAG", "LONG_IN_CONTEXT", "HYDE_RAG"]:
            raise ValueError(
                f"Expected retrieval type to be one of ['DIRECT', 'BASIC_RAG', 'LONG_IN_CONTEXT', 'HYDE_RAG'] but got {retrieval_type}"
            )

        retrieval_type_enum = RetrievalType(retrieval_type)

        assert isinstance(question_document, QuestionDocument), (
            f"Expected question_document to be of type QuestionDocument. Got {type(question_document)}"
        )

        llm_client_to_use = alternative_llm_client or self.default_sync_client

        if not llm_client_to_use:
            raise ValueError(
                "You must set a default LLM Client or pass one in before performing any operations"
            )

        llm_client_enum = llm_client_to_use.enum_name

        questions_for_experiment = question_document.questions

        experiment_manager = self._get_experiment_manager(
            alternative_llm_client=alternative_llm_client,
            alternative_hypothetical_answer_prompt=alternative_prompt_for_hyde.content
            if alternative_prompt_for_hyde is not None
            else None,
        )

        ai_model_for_hyde = (
            ai_model_for_hyde or alternative_llm_client_for_hyde.config.default_model
            if alternative_llm_client_for_hyde
            else None
        )

        al_model_to_use = ai_model_to_use or llm_client_to_use.config.default_model

        experiment_params = ExperimentParams(
            system_prompt=system_prompt,
            experiment_type=experiment_type_enum,
            retrieval_type=retrieval_type_enum,
            input_path=input_store_path,
            output_path=output_store_path,
            questions=questions_for_experiment,
            llm_client_enum=llm_client_enum,
            ai_model_for_experiment=al_model_to_use,
            knowledge_base_identifier=question_document.knowledge_base_identifier,
            alternative_llm_client_for_hyde=alternative_llm_client_for_hyde,
            alternative_prompt_for_hyde=alternative_prompt_for_hyde,
            ai_model_for_hyde=ai_model_for_hyde,
        )

        output = experiment_manager.create_experiment(
            experiment_params=experiment_params
        )
        output.save_to_json()

        return output

    def create_all_inputs_for_experiment(
        self,
        question_document: Union[Path, QuestionDocument],
        experiment_type: Literal["removal", "synthetic"] = "removal",
        base_path: Path = Path("experiments"),
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model_to_use: Optional[str] = None,
        alternative_prompt_for_hyde: Optional[Prompt] = None,
        alternative_llm_client_for_hyde: Optional[SyncLLMClient] = None,
        ai_model_for_hyde: Optional[str] = None,
    ) -> List[ExperimentInputDocument]:
        """
        Creates multiple experiment input documents using different prompt types and retrieval strategies.

        Args:
            question_document: The question document to use for the experiments
            experiment_type: The type of experiment (removal or synthetic)
            base_path: Base directory to store experiment files
            alternative_llm_client: Optional alternative LLM client to use
            ai_model_to_use: Optional AI model to use
            alternative_prompt_for_hyde: Optional alternative prompt for HYDE_RAG
            alternative_llm_client_for_hyde: Optional alternative LLM client for HYDE_RAG
            ai_model_for_hyde: Optional AI model for HYDE_RAG

        Returns:
            List of created experiment input documents
        """
        if isinstance(question_document, Path):
            question_document = QuestionDocument.load_from_json(question_document)

        knowledge_base_id = question_document.knowledge_base_identifier

        # Determine which model to use
        client_to_use = alternative_llm_client or self.default_sync_client
        if not client_to_use:
            raise ValueError(
                "No default client available and no alternative client provided"
            )

        model_to_use = None
        if ai_model_to_use:
            model_to_use = ai_model_to_use
        else:
            model_to_use = client_to_use.config.default_model

        # Set up prompts and allowed retrieval types for each
        prompt_configs = [
            {
                "name": "basic",
                "prompt": Prompt(
                    identifier="basic_llm_prompt",
                    content=PromptManager.basic_llm_prompt,
                ),
                "allowed_retrieval_types": ["DIRECT", "LONG_IN_CONTEXT"],
            },
            {
                "name": "conservative",
                "prompt": Prompt(
                    identifier="conservative_llm_prompt",
                    content=PromptManager.conservative_llm_prompt,
                ),
                "allowed_retrieval_types": [
                    "DIRECT",
                    "BASIC_RAG",
                    "LONG_IN_CONTEXT",
                    "HYDE_RAG",
                ],
            },
            {
                "name": "opinion",
                "prompt": Prompt(
                    identifier="opinion_llm_prompt",
                    content=PromptManager.opinion_llm_prompt,
                ),
                "allowed_retrieval_types": [
                    "DIRECT",
                    "BASIC_RAG",
                    "LONG_IN_CONTEXT",
                    "HYDE_RAG",
                ],
            },
        ]

        experiment_inputs = []

        for config in prompt_configs:
            prompt_name = config["name"]
            system_prompt = config["prompt"]
            allowed_retrieval_types = config["allowed_retrieval_types"]

            for retrieval_type in allowed_retrieval_types:
                # Create paths with all necessary information
                input_path = (
                    base_path
                    / "inputs"
                    / f"{knowledge_base_id}_{experiment_type}_{retrieval_type}_{prompt_name}_{model_to_use}_input.json"
                )
                output_path = (
                    base_path
                    / "outputs"
                    / f"{knowledge_base_id}_{experiment_type}_{retrieval_type}_{prompt_name}_{model_to_use}_output.json"
                )

                # Ensure directories exist
                input_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Create the experiment input
                experiment_input = self.create_experiment_input(
                    question_document=question_document,
                    system_prompt=system_prompt,
                    experiment_type=experiment_type,
                    retrieval_type=retrieval_type,
                    input_store_path=input_path,
                    output_store_path=output_path,
                    alternative_llm_client=alternative_llm_client,
                    ai_model_to_use=ai_model_to_use,
                    alternative_prompt_for_hyde=alternative_prompt_for_hyde,
                    alternative_llm_client_for_hyde=alternative_llm_client_for_hyde,
                    ai_model_for_hyde=ai_model_for_hyde,
                )

                experiment_inputs.append(experiment_input)

        return experiment_inputs

    def run_experiment_sync(
        self,
        experiment_input: ExperimentInputDocument,
    ) -> ExperimentOutputDocument:
        """
        Executes an experiment synchronously using the specified experiment input.

        This function retrieves the appropriate LLM client based on the client enum
        in the experiment metadata, then executes the experiment using the ExperimentManager.
        The result is saved to a JSON file.

        Args:
            experiment_input (ExperimentInputDocument): The input document containing
            the experiment details and metadata.

        Returns:
            ExperimentOutputDocument: The output document containing the results of the experiment.

        Raises:
            ValueError: If the client enum specified in the experiment metadata is not registered.
        """

        llm_client_enum = experiment_input.metadata.client_enum
        try:
            client_to_use = self.get_client(llm_client_enum)

        except KeyError:
            raise ValueError(f"Client with enum {llm_client_enum} is not registered")

        self.logger.info(f"Client to use: {client_to_use}")

        experiment_manager = self._get_experiment_manager(
            alternative_llm_client=client_to_use
        )
        output = experiment_manager.run_experiment_sync(
            experiment=experiment_input,
            client_registry=self.client_registry,
        )
        output.save_to_json()

        return output

    async def run_experiment_async(
        self,
        experiment_input: ExperimentInputDocument,
    ) -> ExperimentOutputDocument:
        """
        Executes an experiment asynchronously using the specified experiment input.

        This function retrieves the appropriate LLM client based on the client enum
        in the experiment metadata, then executes the experiment using the ExperimentManager.
        The result is saved to a JSON file.

        Args:
            experiment_input (ExperimentInputDocument): The input document containing
            the experiment details and metadata.

        Returns:
            ExperimentOutputDocument: The output document containing the results of the experiment.

        Raises:
            ValueError: If the client enum specified in the experiment metadata is not registered.
        """

        llm_client_enum = experiment_input.metadata.client_enum
        try:
            client_to_use = self.get_client(llm_client_enum)

        except KeyError:
            raise ValueError(f"Client with enum {llm_client_enum} is not registered")

        self.logger.info(f"Client to use: {client_to_use}")

        experiment_manager = self._get_experiment_manager(
            alternative_llm_client=client_to_use
        )
        output = await experiment_manager.run_experiment_async(
            experiment=experiment_input,
            client_registry=self.client_registry,
        )
        output.save_to_json()

        return output

    def create_evaluation_spec(
        self,
        evaluation_name: str,
        prompt_identifier: str,
        prompt_content: str,
        evaluation_outcomes: List[str],
        tag_name: str,
        in_context: List[Literal["question", "expected_answer", "context"]] = [
            "question",
            "expected_answer",
            "context",
        ],
        recommended_llm_client_enum: Optional[SyncLLMClientEnum] = None,
        recommended_llm_model: Optional[str] = None,
    ) -> EvaluationSpec:
        """
        Creates an EvaluationSpec object based on the provided parameters.

        Args:
            evaluation_name (str): The name of the evaluation.
            prompt_identifier (str): The identifier for the prompt.
            prompt_content (str): The content of the prompt.
            evaluation_outcomes (List[str]): The possible outcomes of the evaluation.
            tag_name (str): The tag associated with the evaluation.
            in_context (List[Literal["question", "expected_answer", "context"]], optional):
                The parts of the prompt to include in the context. Defaults to [
                    "question", "expected_answer", "context"].
            recommended_llm_client_enum (Optional[SyncLLMClientEnum], optional):
                The recommended LLM client enum for this evaluation. Defaults to
                None, in which case the default client's enum is used.
            recommended_llm_model (Optional[str], optional):
                The recommended LLM model for this evaluation. Defaults to None in which
                case the given client's default model is used, or the default client's model is used.

        Returns:
            EvaluationSpec: The created EvaluationSpec object.
        """

        if not recommended_llm_client_enum and self.default_sync_client:
            recommended_llm_client_enum = self.default_sync_client.enum_name
        else:
            raise ValueError(
                "No default LLM client is available and no client is specified"
            )

        return EvaluationSpec(
            name=evaluation_name,
            prompt=Prompt(content=prompt_content, identifier=prompt_identifier),
            in_context=in_context,
            tag_name=tag_name,
            evaluation_outcomes=evaluation_outcomes,
            recommended_llm_client_enum=recommended_llm_client_enum,
            recommended_llm_model=recommended_llm_model,
        )

    def create_evaluator(
        self,
        evaluation_list: List[EvaluationSpec],
        alternative_llm_client: Optional[SyncLLMClient] = None,
        alternative_llm_model: Optional[str] = None,
    ) -> Evaluator:
        """
        Creates an Evaluator object from the given evaluation specifications.

        Args:
            evaluation_dict (Dict[str, EvaluationSpec]): A dictionary mapping evaluation names to EvaluationSpec objects.
            alternative_llm_client (Optional[SyncLLMClient], optional): The LLM client to use for evaluation. Defaults to None.
            alternative_llm_model (Optional[str], optional): The LLM model to use for evaluation. Defaults to None.

        Returns:
            Evaluator: The created Evaluator object.
        """
        evaluation_dict = {spec.name: spec for spec in evaluation_list}
        evaluator = self._get_evaluator(
            evaluation_spec_dict=evaluation_dict,
            alternative_llm_client=alternative_llm_client,
            model_to_use=alternative_llm_model,
        )

        return evaluator

    def evaluate_experiment(
        self, experiment_output: ExperimentOutputDocument, path_to_store: Path
    ) -> EvaluatedExperimentDocument:
        if not self.evaluator:
            raise ValueError(
                "You must create an evaluator with create_evaluator before evaluating an experiment"
            )

        return self.evaluator.evaluate_document(
            document=experiment_output,
            client_registry=self.client_registry,
            path_to_store=path_to_store,
        )

    async def evaluate_experiment_async(
        self, experiment_output: ExperimentOutputDocument, path_to_store: Path
    ) -> EvaluatedExperimentDocument:
        if not self.evaluator:
            raise ValueError(
                "You must create an evaluator with create_evaluator before evaluating an experiment"
            )

        return await self.evaluator.evaluate_document_async(
            document=experiment_output,
            client_registry=self.client_registry,
            path_to_store=path_to_store,
        )

    def create_samples_to_label(
        self,
        experiment_outputs: List[ExperimentOutputDocument],
        percentage_to_sample: float,
        path_to_store: Path,
    ) -> List[LabeledDataSample]:
        if not 0 < percentage_to_sample <= 1:
            raise ValueError("percentage_to_sample must be between 0 and 1")

        if not path_to_store.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path_to_store}")

        data_labeller = self._get_data_labeller(logger=self.logger)

        return data_labeller.sample_data_stratified(
            experiments=experiment_outputs,
            percentage_to_sample=percentage_to_sample,
            json_path=path_to_store,
        )
