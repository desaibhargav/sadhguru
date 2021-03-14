import streamlit as st
import pandas as pd
import os

from frontend.utils import (
    search_pipeline,
    explore_pipeline,
    process_pipeline,
)
from backend.utils import load_from_cache, save_to_cache
from backend.dataloader import DataLoader
from backend.recommender import Recommender


def main():
    # set page config
    st.set_page_config(
        page_title="Ask Sadhguru",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    # set some variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    state = app_state()

    # set the title
    st.title("Ask Sadhguru")

    # # create the database
    # if not isinstance(state["database"], pd.DataFrame):
    #     database_cache_path = os.path.join(os.getcwd(), "cache", "database.pickle")
    #     if os.path.isfile(database_cache_path):
    #         df = load_from_cache("database")
    #     else:
    #         df = load_database(
    #             os.path.join(
    #                 os.getcwd(),
    #                 "datasets",
    #                 "youtube_scrapped_complete_protocol5.pickle",
    #             ),
    #             save_state=True,
    #         )
    #     state["database"] = df

    if not state["database"]:
        if os.path.isfile(os.path.join(os.getcwd(), "cache", "database2.pickle")):
            database_dict = load_from_cache("database2")
        else:
            database_dict = load_database()
            save_to_cache("database2", database_dict)
        state["database"] = database_dict

    # # fit the database
    # if not isinstance(state["recommender"], Recommender):
    #     recommender_cache_path = os.path.join(
    #         os.getcwd(), "cache", "recommender.pickle"
    #     )
    #     if os.path.isfile(recommender_cache_path):
    #         recommender = load_from_cache("recommender")
    #     else:
    #         with st.spinner("Fitting the database"):
    #             recommender = Recommender()
    #             recommender.fit(corpus=state["database"])
    #     state["recommender"] = recommender
    if not state["recommender"]:
        if os.path.isfile(os.path.join(os.getcwd(), "cache", "recommender2.pickle")):
            recommender = load_from_cache("recommender2")
        else:
            recommender = Recommender(corpus_dict=state["database"])
            recommender.fit()
            save_to_cache("recommender2", recommender)
        state["recommender"] = recommender

    # once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["How does it work?", "Search", "Explore"]
    )
    if app_mode == "How does it work?":
        st.sidebar.success("Pulling up the home page")
        process_pipeline(state["database"])
    elif app_mode == "Search":
        st.sidebar.success("Search selected")
        search_pipeline(state["recommender"])
    elif app_mode == "Explore":
        st.sidebar.success("Explore selected")
        explore_pipeline(state["recommender"])


@st.cache(allow_output_mutation=True, show_spinner=True)
def load_database() -> dict:
    """"""
    youtube_dataset = DataLoader.load_youtube_dataset()
    podcast_dataset = DataLoader.load_podcast_dataset()
    return {"youtube": youtube_dataset, "podcast": podcast_dataset}
    # return create_database(file, save_state=save_state)


@st.cache(allow_output_mutation=True, show_spinner=True)
def app_state() -> dict:
    state = {"recommender": False, "database": False}
    return state


if __name__ == "__main__":
    main()