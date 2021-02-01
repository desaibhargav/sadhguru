import streamlit as st

st.title("Ask Sadhguru")
st.header("Search")
question = st.text_area(
    "Enter your question here", "How to make big decisions in life?"
)
search = st.beta_expander("Recommmendations", expanded=False)
with search:
    for i in range(1, 3):
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.header("Video 1")
            st.video("https://www.youtube.com/watch?v=-2IcOOUqNgI", start_time=8)

        with col2:
            st.header("Video 2")
            st.video("https://www.youtube.com/watch?v=-3rzessN6cI", start_time=6)

        with col3:
            st.header("Video 3")
            st.video("https://www.youtube.com/watch?v=-46JXxFlXoA", start_time=92)

st.header("Explore")
question = st.text_input("Search here", "Sadhguru on Coronavirus pandemic")
explore = st.beta_expander("Recommmendations", expanded=False)
with explore:
    for i in range(1, 3):
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            st.header("Video 1")
            st.video("https://www.youtube.com/watch?v=-2IcOOUqNgI", start_time=8)

        with col2:
            st.header("Video 2")
            st.video("https://www.youtube.com/watch?v=-3rzessN6cI", start_time=6)

        with col3:
            st.header("Video 3")
            st.video("https://www.youtube.com/watch?v=-46JXxFlXoA", start_time=92)
