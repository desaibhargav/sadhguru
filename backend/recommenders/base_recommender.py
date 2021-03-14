import torch
import pandas as pd

from itertools import chain
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class BaseRecommender:
    def __init__(self, corpus: pd.DataFrame, feature_to_column_mapping: dict):
        self.corpus = corpus
        self.feature_to_column_mapping = feature_to_column_mapping

    def _encode(
        self, content: Union[List[str], str], encoder: SentenceTransformer
    ) -> torch.Tensor:
        return encoder.encode(content, convert_to_tensor=True, show_progress_bar=True)

    def _semamtic_search(
        self, query_embedding: torch.Tensor, corpus_str: str, top_k: int
    ) -> List[dict]:
        return util.semantic_search(
            query_embedding, self.corpus_embeddings_dict[corpus_str], top_k=top_k
        ).pop()

    def fit(self, encoder: SentenceTransformer):
        """
        fit the columns from the corpus to be used for recommendations
        """
        columns_to_fit = chain.from_iterable(self.feature_to_column_mapping.values())
        assert (
            pd.Series(columns_to_fit).isin(self.corpus.columns).all()
        ), "column(s) to fit do not exist in the passed corpus [pd.DataFrame object]"
        self.corpus_embeddings_dict = {
            column: self._encode(self.corpus[column].unique(), encoder=encoder)
            for column in columns_to_fit
        }

    def search(
        self,
        question: str,
        encoder: SentenceTransformer,
        cross_encoder: CrossEncoder,
        top_k: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def explore(
        self, query: str, encoder: SentenceTransformer, top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass