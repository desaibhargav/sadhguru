import pandas as pd

from backend.chunker import Chunker


class DataLoader:
    @staticmethod
    def load_podcast_dataset():

        # load podcast data from /datasets
        podcast_data = pd.read_pickle("datasets/podcast_scrapped.pickle")

        # chunk the data into blocks (the fundamental unit of this project)
        chunked_podcast_data = Chunker(
            chunk_by="length", expected_threshold=100, min_tolerable_threshold=75
        ).get_chunks(podcast_data)

        # finally, create and return the dataset
        podcast_dataset = podcast_data.join(chunked_podcast_data).drop(
            columns=["subtitles", "timestamps"]
        )
        podcast_dataset = podcast_dataset.dropna().drop_duplicates(subset=["block"])
        podcast_dataset.block = podcast_dataset.block.apply(lambda x: x.lower())
        return podcast_dataset

    @staticmethod
    def load_youtube_dataset():

        # load youtube data from /datasets
        youtube_data = pd.read_pickle(
            "datasets/youtube_scrapped_complete_protocol5.pickle"
        )

        # chunk the data into blocks (the fundamental unit of this project)
        chunked_youtube_data = Chunker(
            chunk_by="length", expected_threshold=100, min_tolerable_threshold=75
        ).get_chunks(youtube_data)

        # finally, create and return the dataset
        youtube_dataset = youtube_data.join(chunked_youtube_data).drop(
            columns=["subtitles", "timestamps"]
        )
        youtube_dataset = youtube_dataset.dropna()
        youtube_dataset = youtube_dataset.assign(
            video_description=youtube_dataset["video_description"]
            .str.strip()
            .str.split("\n\n")
            .str[0],
            video_title=youtube_dataset["video_title"].str.strip(),
        )
        youtube_dataset = youtube_dataset.drop_duplicates(subset="block")
        return youtube_dataset