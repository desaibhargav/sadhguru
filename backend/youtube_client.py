import operator
import pandas as pd
import numpy as np
import googleapiclient.discovery
import time

from itertools import chain
from functools import reduce
from typing import Union, Callable, Generator, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm


class YouTubeClient:
    """
    Only to scrape data. Not to upload or delete videos.
    """

    def __init__(self, API_KEY: str):
        self.__API_KEY = API_KEY
        self._api_service_name = "youtube"
        self._api_version = "v3"
        self.youtube_metadata_client = googleapiclient.discovery.build(
            self._api_service_name, self._api_version, developerKey=self.__API_KEY
        )
        self.youtube_transcript_client = YouTubeTranscriptApi()
        self.state = {"units_consumed": 0, "daily_quota": 10000}

    def _execute_query(self, query_kind: str, query: str, **query_params) -> Generator:
        """"""
        assert (
            query_kind == "metadata" or query_kind == "transcript"
        ), f"'query_kind' must be one of ['metadata', 'transcript']"
        if query_kind == "metadata":
            resource, action = query.split(".")
            query_params["pageToken"] = query_params.get("pageToken", "")
            has_next_page = True
            while has_next_page:
                response = getattr(
                    getattr(self.youtube_metadata_client, resource.replace("()", ""))(),
                    action,
                )(**query_params).execute()
                query_params["pageToken"] = response.get("nextPageToken", "")
                has_next_page = bool(query_params.get("pageToken", ""))
                self.state["units_consumed"] += query_params.get("maxResults", 1)
                yield response["items"]
        if query_kind == "transcript":
            try:
                response = getattr(self.youtube_transcript_client, query)(
                    **query_params
                )
                time.sleep(8)
                yield response
            except:
                yield [{"duration": np.NaN, "start": np.NaN, "text": np.NaN}]

    def _process_query(self, query_kind: str, response: Generator) -> pd.DataFrame:
        """
        A generic processing function that should apply to any query.
        To achieve more specific processing modify code in the specific public methods.
        """
        assert (
            query_kind == "metadata" or query_kind == "transcript"
        ), f"'query_kind' must be one of ['metadata', 'transcript']"
        df = pd.DataFrame(list(chain.from_iterable(response)))
        if query_kind == "metadata":
            try:
                df.drop(columns=["kind", "etag"], inplace=True)
            except KeyError:
                pass
        if query_kind == "transcript":
            subtitles = df.text.to_list()
            timestamps = list(
                zip(df.start.to_list(), (df.start + df.duration).to_list())
            )
            df = pd.DataFrame({"subtitles": [subtitles], "timestamps": [timestamps]})
        return df

    def _extract_and_add_as_column(
        self, df: pd.DataFrame, extract_dict: dict, clean_up: bool
    ) -> pd.DataFrame:
        """"""
        target_columns = list(
            map(
                operator.itemgetter(0),
                [[x] if not isinstance(x, list) else x for x in extract_dict["from"]],
            )
        )
        assert len(extract_dict.keys()) == 2 and extract_dict.keys() == {
            "extract",
            "from",
        }, "Passed dict should have only two fields: ('extract', 'from')"
        assert len(extract_dict["extract"]) == len(
            extract_dict["from"]
        ), "Fields to be extracted are not equal to the columns specified"
        assert (
            pd.Series(target_columns).isin(df.columns).all()
        ), "Column(s) from which fields are to be extracted, do not exist in the passed pd.DataFrame object"
        for extract, from_column in zip(*extract_dict.values()):
            if type(from_column) == list:
                df[extract] = df[from_column[0]].apply(
                    lambda x: reduce(operator.getitem, from_column[1:] + [extract], x)
                )
            else:
                df[extract] = df[from_column].apply(lambda x: x.get(extract, np.NaN))
        if clean_up:
            df.drop(columns=list(set(target_columns)), inplace=True)
        return df

    def _align(self, *list_of_dfs: pd.DataFrame, on: str, how: str) -> pd.DataFrame:
        """"""
        assert how in [
            "inner",
            "outer",
        ], f"The argument 'how' should be one of [{'inner', 'outer'}]"
        return reduce(lambda x, y: pd.merge(x, y, on=on, how=how), list_of_dfs)

    def _get_channel_upload_id(self, username: str) -> Tuple[str, str]:
        response = self._execute_query(
            query_kind="metadata",
            query="channels().list",
            part="contentDetails",
            forUsername=username,
        )
        metadata = self._process_query(query_kind="metadata", response=response)
        channel_id = metadata.id.to_list().pop()
        channel_upload_id = (
            metadata.contentDetails.apply(lambda x: x["relatedPlaylists"]["uploads"])
            .to_list()
            .pop()
        )
        return channel_id, channel_upload_id

    def _from_playlist_ids(self, *playlist_ids: Union[str, list]) -> pd.DataFrame:
        """"""
        extract_dict = {
            "extract": ["title", "description", "itemCount"],
            "from": ["snippet", "snippet", "contentDetails"],
        }
        metadata = pd.concat(
            [
                self._process_query(
                    query_kind="metadata",
                    response=self._execute_query(
                        query_kind="metadata",
                        query="playlists().list",
                        id=playlist_id,
                        part="contentDetails, snippet",
                    ),
                )
                for playlist_id in tqdm(playlist_ids)
            ]
        )
        metadata = self._extract_and_add_as_column(
            metadata, extract_dict, clean_up=True
        )
        metadata.rename(
            columns={
                "id": "playlistId",
                "title": "playlist_title",
                "description": "playlist_description",
            },
            inplace=True,
        )
        return metadata

    def _from_video_ids(self, *video_ids: Union[str, list]) -> pd.DataFrame:
        """"""
        extract_dict = {
            "extract": [
                "definition",
                "defaultAudioLanguage",
                "publishedAt",
                "description",
                "title",
                "tags",
                "url",
                "commentCount",
                "dislikeCount",
                "favoriteCount",
                "likeCount",
                "viewCount",
            ],
            "from": [
                "contentDetails",
                "snippet",
                "snippet",
                "snippet",
                "snippet",
                "snippet",
                ["snippet", "thumbnails", "high"],
                "statistics",
                "statistics",
                "statistics",
                "statistics",
                "statistics",
            ],
        }
        metadata = pd.concat(
            [
                self._process_query(
                    query_kind="metadata",
                    response=self._execute_query(
                        query_kind="metadata",
                        query="videos().list",
                        id=video_id,
                        part="snippet, contentDetails, statistics",
                    ),
                )
                for video_id in tqdm(video_ids)
            ]
        )
        metadata = self._extract_and_add_as_column(
            metadata, extract_dict, clean_up=True
        )
        metadata.rename(
            columns={
                "id": "videoId",
                "title": "video_title",
                "description": "video_description",
            },
            inplace=True,
        )
        return metadata

    def from_playlist(self, *playlists: Union[str, list]) -> pd.DataFrame:
        """"""
        is_url = (
            lambda playlist: True
            if playlist.startswith("https://www.youtube.com/watch?")
            else False
        )
        extract_playlist_id = lambda playlist: playlist[playlist.find("&list=") + 6 :]
        playlist_ids = [
            extract_playlist_id(playlist) if is_url(playlist) else playlist
            for playlist in playlists
        ]
        playlist_metadata = self._from_playlist_ids(*playlist_ids)
        extract_dict = {
            "extract": ["playlistId", "videoId"],
            "from": ["snippet", "contentDetails"],
        }
        metadata = pd.concat(
            [
                self._process_query(
                    query_kind="metadata",
                    response=self._execute_query(
                        query_kind="metadata",
                        query="playlistItems().list",
                        playlistId=playlist_id,
                        part="snippet, contentDetails",
                        maxResults=50,
                    ),
                )
                for playlist_id in tqdm(playlist_ids)
            ]
        )
        metadata = self._extract_and_add_as_column(
            metadata, extract_dict, clean_up=True
        )
        metadata.drop(columns=["id"], inplace=True)
        video_metadata = self._from_video_ids(*metadata.videoId.to_list())
        transcript = pd.concat(
            [
                self._process_query(
                    query_kind="transcript",
                    response=self._execute_query(
                        query_kind="transcript",
                        query="get_transcript",
                        video_id=video_id,
                    ),
                )
                for video_id in tqdm(video_metadata.videoId.values)
            ]
        )
        transcript["videoId"] = video_metadata.videoId.to_list()

        # merge on 'videoId', 'playlistId' and return
        video_data_merged = self._align(
            metadata, transcript, video_metadata, on="videoId", how="inner"
        )
        playlist_data_merged = self._align(
            video_data_merged, playlist_metadata, on="playlistId", how="outer"
        ).drop_duplicates(subset="videoId")
        dataset = playlist_data_merged.set_index(
            [
                "playlistId",
                "itemCount",
                "playlist_title",
                "playlist_description",
                "videoId",
            ]
        )
        return dataset

    def from_channel(self, username: str) -> pd.DataFrame:
        """"""
        channel_id, channel_upload_id = self._get_channel_upload_id(username=username)
        print(f"Scrapping videos for username: {username} [channel ID: {channel_id}]")
        dataset = self.from_playlist(channel_upload_id)
        return dataset
