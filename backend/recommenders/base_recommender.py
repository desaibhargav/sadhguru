import torch
import pandas as pd


from typing import List, Union, Tuple
from backend.utils import save_to_cache
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class BaseRecommender:
    def __init__(self):
        self.encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-electra-base")

    def _encode(self, content: Union[List[str], str], verbosity: bool) -> torch.Tensor:
        return self.encoder.encode(
            content, convert_to_tensor=True, show_progress_bar=verbosity
        )

    def _semamtic_search(
        self, query_embedding: torch.Tensor, corpus_str: str, top_k: int
    ) -> List[dict]:
        return util.semantic_search(
            query_embedding, self.corpus_embeddings_dict[corpus_str], top_k=top_k
        ).pop()

    def fit(self, corpus: pd.DataFrame, columns: List[str], save_state: bool):
        """
        fit the corpuses to be used for recommendations
        """
        assert (
            pd.Series(columns).isin(corpus.columns).all()
        ), "columns to fit do not exist in the passed pd.DataFrame object"
        self.corpus = corpus
        self.corpus_embeddings_dict = {
            column: self._encode(corpus[column].unique(), verbosity=True)
            for column in columns
        }
        if save_state:
            save_to_cache("recommender", self)

    def search(
        self, question: str, corpus: str, top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def explore(
        self, query: str, corpus: List[str], top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass