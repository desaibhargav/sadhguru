import streamlit as st
import pandas as pd
import pickle
import sys
import os

from frontend.utils import create_database
from backend.chunker import Chunker
from backend.recommender import Recommender


def main():
    # set the title
    st.title("Ask Sadhguru")

    # create the database
    st.info(
        "Reading the dataset, this will be cached for future runs", show_spinner=True
    )
    df = create_database(
        os.path.join(
            os.getcwd(), "datasets", "youtube_scrapped_complete_protocol5.pickle"
        )
    )

    # download the models required to fire up the recommender
    st.info(
        "Downloading models, these will be cached for future runs", show_spinner=True
    )
    recommender = setup_recommender()

    # fit the database
    st.info("Fitting the database on the models", show_spinner=True)
    recommender = fit(df, recommender)

    # once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Search", "Explore"]
    )
    if app_mode == "Show instructions":
        st.sidebar.success("Option 1 selected")
    elif app_mode == "Show the source code":
        st.sidebar.success("Option 2 selected")
        # readme_text.empty()
        # st.code(get_file_content_as_string("streamlit_app.py"))
    elif app_mode == "Run the app":
        st.sidebar.success("Option 3 selected")
        # readme_text.empty()
        # run_the_app()


# st.title("Ask Sadhguru")
# st.header("Search")
# question = st.text_area(
#     "Enter your question here", "How to make big decisions in life?"
# )
# search = st.beta_expander("Recommmendations", expanded=False)
# with search:
#     for i in range(1, 3):
#         col1, col2, col3 = st.beta_columns(3)
#         with col1:
#             st.header("Video 1")
#             st.video("https://www.youtube.com/watch?v=-2IcOOUqNgI", start_time=8)

#         with col2:
#             st.header("Video 2")
#             st.video("https://www.youtube.com/watch?v=-3rzessN6cI", start_time=6)

#         with col3:
#             st.header("Video 3")
#             st.video("https://www.youtube.com/watch?v=-46JXxFlXoA", start_time=92)

# st.header("Explore")
# question = st.text_input("Search here", "Sadhguru on Coronavirus pandemic")
# explore = st.beta_expander("Recommmendations", expanded=False)
# with explore:
#     for i in range(1, 3):
#         col1, col2, col3 = st.beta_columns(3)
#         with col1:
#             st.header("Video 1")
#             st.video("https://www.youtube.com/watch?v=-2IcOOUqNgI", start_time=8)

#         with col2:
#             st.header("Video 2")
#             st.video("https://www.youtube.com/watch?v=-3rzessN6cI", start_time=6)

#         with col3:
#             st.header("Video 3")
#             st.video("https://www.youtube.com/watch?v=-46JXxFlXoA", start_time=92)


@st.cache
def load_database(file: str) -> pd.DataFrame:
    """"""
    return create_database(file)


@st.cache
def setup_recommender() -> Recommender:
    return Recommender()


@st.cache
def fit(df: pd.DataFrame, recommender: Recommender) -> Recommender:
    recommender.fit(
        blocks=df.block[:10].to_list(),
        video_titles=df.video_title.unique()[:10],
        video_descriptions=df.video_description.unique()[:10],
    )
    return recommender


if __name__ == "__main__":
    main()