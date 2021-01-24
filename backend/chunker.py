import operator
import pandas as pd


class Chunker:
    def __init__(self, expected_length: int, min_tolerable_length: int):
        assert (
            type(expected_length) and type(min_tolerable_length) == "int"
        ), f"'expected_length' and 'min_tolerable_length' should be of type <int>"
        self.expected_length = expected_length
        self.min_tolerable_length = min_tolerable_length

    def generate_blocks(
        self,
        subtitles: list,
        timestamps: list,
        expected_length: int,
        min_tolerable_length: int,
    ) -> str:
        """"""
        state = {"block": list(), "length_of_block": 0, "covered_length": 0}
        total_length = sum([len(subtitle.strip().split()) for subtitle in subtitles])
        start_times, end_times = zip(*timestamps)
        for idx, subtitle in enumerate(subtitles):
            state["block"].append(subtitle)
            state["length_of_block"] += len(subtitle.strip().split())
            state["covered_length"] += len(subtitle.strip().split())
            if (
                state["length_of_block"] >= expected_length
                and (total_length - state["covered_length"]) >= min_tolerable_length
            ):
                yield (
                    " ".join(state["block"]),
                    start_times[subtitles.index(state["block"][0])],
                    end_times[idx],
                )
                state["block"] = list()
                state["length_of_block"] = 0
            elif (total_length - state["covered_length"]) <= min_tolerable_length:
                state["block"].extend(subtitles[idx:])
                yield (
                    " ".join(state["block"]),
                    start_times[subtitles.index(state["block"][0])],
                    end_times[-1],
                )
                break
            else:
                continue
