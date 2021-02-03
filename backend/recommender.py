import pandas as pd
import pickle
import os

from typing import List
from backend.utils import save_to_cache
from sentence_transformers import SentenceTransformer, CrossEncoder, util


class Recommender:
    def __init__(self):
        self.encoder = SentenceTransformer("paraphrase-distilroberta-base-v1")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-electra-base")

    def fit(self, save_state: bool, **corpus: List[str]):
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
        if save_state:
            save_to_cache("recommender", self)

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
        return pd.DataFrame(hits)

    def explore(self, query: str, top_k: int) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def format_for_frontend(df: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
        hits = hits.loc[hits["cross-score"] >= 0.25]
        hits = hits.assign(
            video_link=hits.corpus_id.apply(
                lambda x: f"https://www.youtube.com/watch?v={df.index.get_level_values(0)[x]}"
            ),
            start=hits.corpus_id.apply(lambda x: df.start_time.iloc[x]),
            end=hits.corpus_id.apply(lambda x: df.end_time.iloc[x]),
        ).sort_values("start")
        recommendations = (
            hits.groupby("video_link", as_index=False)
            .agg({"start": "min", "end": "max", "cross-score": "max"})
            .sort_values("cross-score", ascending=False)
        )
        return recommendations
