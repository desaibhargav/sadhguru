import pandas as pd
import torch

from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class Recommender:
    def __init__(self):
        self.encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-electra-base")

    def fit(self, **corpus: List[str]) -> Optional[Tuple[List[str], torch.Tensor]]:
        """
        fit the corpuses to be used for recommendations
        """
        self.corpus_dict = corpus
        self.corpus_embeddings_dict = {
            key: self.encoder.encode(
                value, convert_to_tensor=True, show_progress_bar=True
            )
            for key, value in corpus.items()
        }

    def search(self, question: str, corpus: str, top_k: int) -> pd.DataFrame:
        """
        semantic search
        """
        assert (
            corpus in self.corpus_dict
        ), "Corpus not found, please fit the corpus first using the .fit() call"
        question_embedding = self.encoder.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(
            question_embedding, self.corpus_embeddings_dict[corpus], top_k=top_k
        ).pop()

        # now, score all retrieved passages with the cross_encoder
        cross_inp = [
            [question, self.corpus_dict[corpus][hit["corpus_id"]]] for hit in hits
        ]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]["cross-score"] = cross_scores[idx]
            hits[idx]["snippet"] = self.corpus_dict[corpus][
                hits[idx]["corpus_id"]
            ].replace("\n", " ")
        hits = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
        return hits

    def explore(self, query: str, top_k: int) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def format_for_frontend(df: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def format_for_web(df: pd.DataFrame):
        raise NotImplementedError