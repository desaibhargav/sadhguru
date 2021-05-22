import torch
import pandas as pd

from typing import Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.recommenders.base_recommender import BaseRecommender


class BlogRecommender(BaseRecommender):
    def __init__(self, corpus: pd.DataFrame, feature_to_column_mapping: dict = dict()):
        if not feature_to_column_mapping:
            feature_to_column_mapping = {
                "search": "block",
                "explore": ["video_title", "video_description"],
            }
        super(BlogRecommender, self).__init__(
            corpus=corpus, feature_to_column_mapping=feature_to_column_mapping
        )

    def search(
        self,
        question: str,
        encoder: SentenceTransformer,
        cross_encoder: CrossEncoder,
        top_k: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        semantic search
        """
        raise NotImplementedError(
            "Just a placeholder for now, implementation is ongoing and will be added in the next release"
        )

    def explore(
        self,
        query: str,
        encoder: SentenceTransformer,
        top_k: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError(
            "Just a placeholder for now, implementation is ongoing and will be added in the next release"
        )
