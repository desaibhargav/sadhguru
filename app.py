import streamlit as st
import pandas as pd
import os

from frontend.utils import create_database, search_pipeline, explore_pipeline
from backend.utils import load_from_cache
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
    readme_text = st.markdown(open(os.path.join(os.getcwd(), "README.md")).read())

    # create the database
    if not isinstance(state["database"], pd.DataFrame):
        database_cache_path = os.path.join(os.getcwd(), "cache", "database.pickle")
        if os.path.isfile(database_cache_path):
            df = load_from_cache("database")
        else:
            df = load_database(
                os.path.join(
                    os.getcwd(),
                    "datasets",
                    "youtube_scrapped_complete_protocol5.pickle",
                ),
                save_state=True,
            )
        state["database"] = df

    # fit the database
    if not isinstance(state["recommender"], Recommender):
        recommender_cache_path = os.path.join(
            os.getcwd(), "cache", "recommender.pickle"
        )
        if os.path.isfile(recommender_cache_path):
            recommender = load_from_cache("recommender")
        else:
            recommender = Recommender()
            recommender.fit(
                blocks=state["database"].block.to_list(),
                video_titles=state["database"].video_title.unique(),
                video_descriptions=state["database"].video_description.unique(),
                save_state=True,
            )
        state["recommender"] = recommender

    # once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Search", "Explore"]
    )
    if app_mode == "Show instructions":
        st.sidebar.success("Pulling up the instruction page")
    elif app_mode == "Search":
        readme_text.empty()
        st.sidebar.success("Search selected")
        search_pipeline(state["recommender"], state["database"])
    elif app_mode == "Explore":
        readme_text.empty()
        st.sidebar.success("Explore selected")
        explore_pipeline()


@st.cache(allow_output_mutation=True, show_spinner=True)
def load_database(file: str, save_state: bool) -> pd.DataFrame:
    """"""
    return create_database(file, save_state=save_state)


@st.cache(allow_output_mutation=True, show_spinner=True)
def app_state() -> dict:
    state = {"recommender": False, "database": False}
    return state


if __name__ == "__main__":
    main()