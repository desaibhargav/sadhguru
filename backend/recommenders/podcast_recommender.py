import pandas as pd

from typing import Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.recommenders.base_recommender import BaseRecommender


class PodcastRecommender(BaseRecommender):
    def __init__(self, corpus: pd.DataFrame, feature_to_column_mapping: dict = dict()):
        if not feature_to_column_mapping:
            feature_to_column_mapping = {
                "search": "block",
                "explore": ["title", "excerpt"],
            }
        super(PodcastRecommender, self).__init__(
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
        column = self.feature_to_column_mapping["search"]
        assert (
            column in self.corpus_embeddings_dict
        ), f"Embeddings for [{column}] not found, please fit [{column}] first using the .fit() call"
        question_embedding = self._encode(question, encoder)
        hits = self._semamtic_search(question_embedding, column, top_k)

        # score all retrieved passages with the cross_encoder
        cross_inp = [[question, self.corpus[column][hit["corpus_id"]]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        # sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]["cross-score"] = cross_scores[idx]
            hits[idx]["snippet"] = self.corpus[column][hits[idx]["corpus_id"]].replace(
                "\n", " "
            )

        # return hits and recommendations
        hits = (
            pd.DataFrame(hits)
            .sort_values("cross-score", ascending=False)
            .query("`cross-score` >= 0.15")
        )
        recommendations = hits.assign(
            podcast_link=hits.corpus_id.apply(
                lambda x: self.corpus.loc[
                    self.corpus.index.get_level_values(0)[x]
                ].audio_url.to_list()[0]
            ),
            snippet=hits.corpus_id.apply(lambda x: self.corpus.block.iloc[x]),
            start=hits.corpus_id.apply(lambda x: self.corpus.start_time.iloc[x]),
            end=hits.corpus_id.apply(lambda x: self.corpus.end_time.iloc[x]),
        ).sort_values("start")
        recommendations = (
            recommendations.groupby("podcast_link", as_index=False)
            .agg({"start": "min", "end": "max", "cross-score": "max"})
            .sort_values("cross-score", ascending=False)
        )
        return (hits, recommendations)

    def explore(
        self, query: str, encoder: SentenceTransformer, top_k: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        columns = self.feature_to_column_mapping["explore"]
        assert all(column in self.corpus_embeddings_dict for column in columns)

        # get hits
        question_embedding = self._encode(query, encoder)
        hits = pd.concat(
            [
                pd.DataFrame(
                    self._semamtic_search(question_embedding, column, top_k=top_k)
                )
                for column in columns
            ],
            axis=1,
        )

        # format hits
        hits.columns = (" score ".join(columns) + " score ").strip().split()
        hits_corpus = pd.DataFrame(hits.loc[:, columns].stack()).reset_index()
        hits_score = pd.DataFrame(hits.loc[:, ["score"]].stack()).reset_index(drop=True)
        hits = pd.concat([hits_corpus, hits_score], axis=1).drop(columns=["level_0"])
        hits.columns = ["type", "corpus_id", "score"]
        hits.sort_values("score", ascending=False, inplace=True)

        # return hits and recommendations
        recommendations = hits.assign(
            podcast_link=hits.apply(
                lambda x: self.corpus.loc[
                    self.corpus.loc[
                        self.corpus[x["type"]]
                        == self.corpus[x["type"]].unique()[x["corpus_id"]]
                    ]
                    .index.get_level_values(0)
                    .to_list()
                    .pop()
                ]
                .audio_url.to_list()
                .pop(),
                axis=1,
            ),
            share_link=hits.apply(
                lambda x: self.corpus.loc[
                    self.corpus[x["type"]]
                    == self.corpus[x["type"]].unique()[x["corpus_id"]]
                ]
                .share_url.to_list()
                .pop(),
                axis=1,
            ),
            video_title=hits.apply(
                lambda x: self.corpus.loc[
                    self.corpus[x["type"]]
                    == self.corpus[x["type"]].unique()[x["corpus_id"]]
                ]
                .title.to_list()
                .pop(),
                axis=1,
            ),
            video_description=hits.apply(
                lambda x: self.corpus.loc[
                    self.corpus[x["type"]]
                    == self.corpus[x["type"]].unique()[x["corpus_id"]]
                ]
                .excerpt.to_list()
                .pop(),
                axis=1,
            ),
        )
        recommendations = recommendations.drop_duplicates(subset=["podcast_link"])
        return (hits, recommendations)