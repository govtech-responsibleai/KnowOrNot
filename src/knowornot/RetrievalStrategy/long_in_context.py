from . import BaseRetrievalStrategy, RetrievalType
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPairFinal, QAWithContext
from typing import List, Optional
import numpy as np
import logging


class LongInContextStrategy(BaseRetrievalStrategy):
    def __init__(
        self, default_client: SyncLLMClient, logger: logging.Logger, closest_k: int = 5
    ):
        super().__init__(
            default_client=default_client, logger=logger, closest_k=closest_k
        )

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
        alterative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=remaining_qa,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPairFinal,
        question_list: List[QAPairFinal],
        embeddings: np.ndarray,
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer="",
            context_questions=question_list,
        )

    @property
    def experiment_type(self):
        return RetrievalType.LONG_IN_CONTEXT
