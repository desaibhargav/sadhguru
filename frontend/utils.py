import os
import pandas as pd
import numpy as np
import streamlit as st

from pathlib import Path
from backend.recommender import Recommender

# def create_database(file: str, save_state: bool) -> pd.DataFrame:
#     """"""
#     assert isinstance(
#         file, str
#     ), "please pass the path to a pickled pd.DataFrame object"
#     try:
#         database = pd.read_pickle(file)
#         chunked = Chunker(
#             chunk_by="length", expected_threshold=100, min_tolerable_threshold=75
#         ).get_chunks(database)
#         database_chunked = database.join(chunked).drop(
#             columns=["subtitles", "timestamps"]
#         )
#         database_chunked.dropna(inplace=True)
#         database_chunked = database_chunked.assign(
#             video_description=database_chunked["video_description"]
#             .str.strip()
#             .str.split("\n\n")
#             .str[0],
#             video_title=database_chunked["video_title"].str.strip(),
#         )
#         database_chunked = database_chunked.drop_duplicates(subset="block")
#         if save_state:
#             save_to_cache("database", database_chunked)
#         return database_chunked
#     except pickle.UnpicklingError:
#         sys.exit("The passed file does not point to a pickled pd.DataFrame object")


def render_recommendations_grid(
    recommendations: pd.DataFrame, mode: str, **grid_specs: int
):
    expander = st.beta_expander("Recommmendations", expanded=True)
    rows = min(
        int(len(recommendations) / grid_specs.get("rows", 3)), grid_specs.get("rows", 3)
    )
    with expander:
        grid_pointer = 0
        for row in range(rows):
            columns = st.beta_columns(grid_specs.get("columns", 3))
            for column in columns:
                with column:
                    if mode == "search":
                        st.header(
                            f"{round(recommendations['cross-score'].iloc[grid_pointer] * 100, 2)}% :heart:"
                        )
                        st.video(
                            recommendations["video_link"].iloc[grid_pointer],
                            start_time=int(recommendations["start"].iloc[grid_pointer]),
                        )
                    else:
                        st.subheader(
                            f"{recommendations['video_title'].iloc[grid_pointer][:30]}..."
                        )
                        st.video(recommendations["video_link"].iloc[grid_pointer])
                grid_pointer += 1


def search_pipeline(recommender: Recommender):
    st.header("Search")
    question = st.text_area(
        "Enter your question here",
        "How to be confident while taking big decisions in life?",
    )
    if st.button("Search"):
        with st.spinner("Searching the database"):
            results_dict = recommender.search(question=question, top_k=100)
            # render_recommendations_grid(
            #     recommendations, mode="search", columns=3, rows=3
            # )
            st.dataframe(results_dict["youtube"]["hits"])
            st.dataframe(results_dict["youtube"]["recommendations"])
            st.dataframe(results_dict["podcast"]["hits"])
            st.dataframe(results_dict["podcast"]["recommendations"])


def explore_pipeline(recommender: Recommender):
    st.header("Explore")
    query = st.text_input("Search here", "love and relationships")
    if st.button("Explore"):
        with st.spinner("Searching the database"):
            results_dict = recommender.explore(query=query, top_k=10)
            # render_recommendations_grid(
            #     recommendations, mode="explore", columns=3, rows=3
            # )
            st.dataframe(results_dict["youtube"]["hits"])
            st.dataframe(results_dict["youtube"]["recommendations"])
            st.dataframe(results_dict["podcast"]["hits"])
            st.dataframe(results_dict["podcast"]["recommendations"])


def process_pipeline(database: pd.DataFrame):
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
        st.dataframe(database.droplevel(level=[1, 2, 3, 4]).reset_index().head())
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

    # st.markdown(open(os.path.join(os.getcwd(), "README.md")).read())
    # raise NotImplementedError