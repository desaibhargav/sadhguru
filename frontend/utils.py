import os
import pandas as pd
import numpy as np
import streamlit as st

from pathlib import Path
from backend.recommender import Recommender


class Grid:
    def __init__(self, rows: int, columns: int, mode: str) -> None:
        self.rows = rows
        self.columns = columns
        self.mode = mode

    def _render_youtube(self, recommendations: pd.DataFrame) -> None:
        num_rows = min(
            int(len(recommendations) / self.rows),
            self.rows,
        )
        youtube_expander = st.beta_expander("Videos for you", expanded=True)
        with youtube_expander:
            grid_pointer = 0
            for _ in range(num_rows):
                columns = st.beta_columns(self.columns)
                for column in columns:
                    with column:
                        if self.mode == "Search":
                            st.header(
                                f"{round(recommendations['cross-score'].iloc[grid_pointer] * 100, 2)}% :heart:"
                            )
                            st.video(
                                recommendations["video_link"].iloc[grid_pointer],
                                start_time=int(
                                    recommendations["start"].iloc[grid_pointer]
                                ),
                            )
                        else:
                            st.subheader(
                                f"{recommendations['video_title'].iloc[grid_pointer][:35]}..."
                            )
                            st.video(recommendations["video_link"].iloc[grid_pointer])
                        grid_pointer += 1

    def _render_podcast(self, recommendations: pd.DataFrame) -> None:
        num_rows = min(
            int(len(recommendations) / self.rows),
            self.rows,
        )
        podcast_expander = st.beta_expander("Podcasts for you", expanded=True)
        with podcast_expander:
            grid_pointer = 0
            for _ in range(num_rows):
                columns = st.beta_columns(self.columns)
                for column in columns:
                    with column:
                        if self.mode == "Search":
                            st.header(
                                f"{round(recommendations['cross-score'].iloc[grid_pointer] * 100, 2)}% :heart:"
                            )
                            st.audio(
                                recommendations["podcast_link"].iloc[grid_pointer],
                                start_time=int(
                                    recommendations["start"].iloc[grid_pointer]
                                ),
                            )
                        else:
                            st.subheader(
                                f"{recommendations['video_title'].iloc[grid_pointer][:35]}..."
                            )
                            st.audio(recommendations["podcast_link"].iloc[grid_pointer])
                        grid_pointer += 1

    def _render_blogs(self, recommendations: pd.DataFrame) -> None:
        raise NotImplementedError

    def render(self, results_dict: dict, filters: list) -> None:
        if "youtube" in filters:
            recommendations = results_dict["youtube"]["recommendations"]
            self._render_youtube(recommendations)
        if "podcast" in filters:
            recommendations = results_dict["podcast"]["recommendations"]
            self._render_podcast(recommendations)
        if not filters:
            st.write("**Oops, nothing to display**")


def experimental_search_pipeline(recommender: Recommender):
    st.header("Search")
    question = st.text_area(
        "Enter your question here",
        "How to be confident while taking big decisions in life?",
    )
    options = st.multiselect(
        "Filter by media type",
        ["youtube", "podcast"],
        ["youtube", "podcast"],
    )
    if st.button("Search"):
        with st.spinner("Searching the database"):
            search_results = recommender.search(question=question, top_k=200)
            Grid(rows=3, columns=3, mode="Search").render(search_results, options)


def experimental_explore_pipeline(recommender: Recommender):
    st.header("Explore")
    query = st.text_input("Search here", "meditation and yoga")
    options = st.multiselect(
        "Filter by media type",
        ["youtube", "podcast"],
        ["youtube", "podcast"],
    )
    if st.button("Explore"):
        with st.spinner("Searching the database"):
            results_dict = recommender.explore(query=query, top_k=10)
            Grid(rows=3, columns=3, mode="Explore").render(results_dict, options)


def process_pipeline(database: dict):
    data_expander = st.beta_expander("What was the data used?", expanded=True)
    with data_expander:
        raw_dataset = pd.read_pickle(
            os.path.join(
                os.getcwd(), "datasets", "youtube_scrapped_complete_protocol5.pickle"
            )
        )
        raw_dataset.subtitles = raw_dataset.subtitles.apply(
            lambda x: np.NaN if isinstance(x[0], float) else x
        )
        st.write(
            "The data collected from the channel using `youtube_client.py` is shown below:"
        )
        st.dataframe(
            raw_dataset.droplevel(level=[0, 1, 2, 3]).dropna().reset_index().head()
        )
        st.write(
            "The subtitle for each video is broken down into blocks of fixed length using `chunker.py`, the data that results looks like:"
        )
        st.dataframe(
            database["youtube"].droplevel(level=[1, 2, 3, 4]).reset_index().head()
        )
        data_markdown = Path(
            os.path.join(os.getcwd(), "frontend", "what_data_do_we_use.md")
        ).read_text()
        st.markdown(data_markdown, unsafe_allow_html=True)
    search_expander = st.beta_expander("How does Search work?", expanded=False)
    with search_expander:
        data_markdown = Path(
            os.path.join(os.getcwd(), "frontend", "how_does_search_work.md")
        ).read_text()
        st.markdown(data_markdown, unsafe_allow_html=True)
    explore_expander = st.beta_expander("How does Explore work?", expanded=False)
    with explore_expander:
        data_markdown = Path(
            os.path.join(os.getcwd(), "frontend", "how_does_explore_work.md")
        ).read_text()
        st.markdown(data_markdown, unsafe_allow_html=True)
    model_expander = st.beta_expander("What were the models used?", expanded=False)
    with model_expander:
        data_markdown = Path(
            os.path.join(os.getcwd(), "frontend", "what_were_the_models_used.md")
        ).read_text()
        st.markdown(data_markdown, unsafe_allow_html=True)