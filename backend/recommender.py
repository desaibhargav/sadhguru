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

    def fit(self, save_state, **corpus: List[str]):
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
        video_id_hits = df.iloc[hits.corpus_id].index.get_level_values(level=0)
        video_url_hits = [
            f"https://www.youtube.com/watch?v={video_id}" for video_id in video_id_hits
        ]
        start_time = df.start_time.iloc[hits.corpus_id]
        end_time = df.end_time.iloc[hits.corpus_id]
        recommendations = pd.DataFrame(
            {"url": video_url_hits, "start": start_time, "end": end_time}
        ).sort_values("start")
        recommendations = recommendations.groupby("url", as_index=False).agg(
            {"start": "min", "end": "max"}
        )
        return recommendations
