import pandas as pd
import numpy as np

from typing import Generator


class Chunker:
    def __init__(self, chunk_by: str, **kwargs: int):
        assert chunk_by in [
            "time",
            "length",
        ], "'chunk_by' must be one of ['time', 'length']"
        self.chunk_by = chunk_by
        if self.chunk_by == "time":
            self.expected_threshold = kwargs.get("expected_threshold", 60)
            self.min_tolerable_threshold = kwargs.get("min_tolerable_threshold", 45)
        if self.chunk_by == "length":
            self.expected_threshold = kwargs.get("expected_threshold", 225)
            self.min_tolerable_threshold = kwargs.get("min_tolerable_threshold", 150)

    def _chunk_by_time(
        self, video_id: str, subtitles: list, timestamps: list
    ) -> Generator:
        """"""
        raise NotImplementedError("Implementation in progress")

    def _chunk_by_length(
        self, video_id: str, subtitles: list, timestamps: list
    ) -> Generator:
        """"""
        state = {"block": list(), "length_of_block": 0, "covered_length": 0}
        pack = lambda **params: params
        try:
            total_length = sum(
                [len(subtitle.strip().split()) for subtitle in subtitles]
            )
            start_times, end_times = zip(*timestamps)
            for idx, subtitle in enumerate(subtitles):
                state["block"].append(subtitle)
                state["length_of_block"] += len(subtitle.strip().split())
                state["covered_length"] += len(subtitle.strip().split())
                if (
                    state["length_of_block"] >= self.expected_threshold
                    and (total_length - state["covered_length"])
                    >= self.min_tolerable_threshold
                ):
                    yield pack(
                        videoId=video_id,
                        block=" ".join(state["block"]),
                        length_of_block=sum(
                            len(line.split()) for line in state["block"]
                        ),
                        start_time=start_times[subtitles.index(state["block"][0])],
                        end_time=end_times[idx],
                    )
                    state["block"] = list()
                    state["length_of_block"] = 0
                elif (
                    total_length - state["covered_length"]
                ) <= self.min_tolerable_threshold:
                    state["block"].extend(subtitles[idx:])
                    yield pack(
                        videoId=video_id,
                        block=" ".join(state["block"]),
                        length_of_block=sum(
                            len(line.split()) for line in state["block"]
                        ),
                        start_time=start_times[subtitles.index(state["block"][0])],
                        end_time=end_times[-1],
                    )
                    break
                else:
                    continue
        except AttributeError:
            yield pack(
                videoId=video_id,
                block=np.NaN,
                length_of_block=np.NaN,
                start_time=np.NaN,
                end_time=np.NaN,
            )

    def get_chunks(self, scapped_df: pd.DataFrame) -> pd.DataFrame:
        """"""
        make_df = lambda blocks: pd.DataFrame(list(blocks))
        index_df = (
            lambda df: df.reset_index()
            .rename(columns={"index": "block_number"})
            .set_index(["videoId", "block_number"])
        )
        if self.chunk_by == "time":
            df = pd.concat(
                [
                    make_df(self._chunk_by_time(video_id, subtitles, timestamps))
                    for video_id, subtitles, timestamps in zip(
                        scapped_df.index.get_level_values("videoId"),
                        scapped_df.subtitles,
                        scapped_df.timestamps,
                    )
                ]
            )
        if self.chunk_by == "length":
            df = pd.concat(
                [
                    make_df(self._chunk_by_length(video_id, subtitles, timestamps))
                    for video_id, subtitles, timestamps in zip(
                        scapped_df.index.get_level_values("videoId"),
                        scapped_df.subtitles,
                        scapped_df.timestamps,
                    )
                ]
            )
        return index_df(df)
