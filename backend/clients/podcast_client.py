import os
import math
import torch
import sndhdr
import librosa
import operator
import requests

import numpy as np
import pandas as pd

from typing import Iterator, List
from pydub import AudioSegment
from tqdm import tqdm
from pydub.exceptions import CouldntDecodeError
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC


class PodcastAPIClient:
    def __init__(self):
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )

    def _check_file_type(self, file_name: str) -> str:
        try:
            return sndhdr.what(file_name).filetype
        except AttributeError:
            return "Unsupported filetype"

    def _convert_mp3_to_wav(self, mp3_content: bytes, file_name: str):
        with open(file_name, "wb") as f:
            f.write(mp3_content)
            f.close()
        try:
            sound = AudioSegment.from_mp3(file_name)
            sound.export(file_name, format="wav")
        except CouldntDecodeError:
            pass

    def _stream_chunks(
        self,
        file_name: str,
        sample_rate_khz: int,
        chunk_duration: int,
        overlap_duration: int,
    ) -> Iterator[dict]:
        try:
            file_type = self._check_file_type(file_name)
            assert file_type == "wav", f"[{file_name}] is not a .wav"
        except AssertionError:
            return list()
        speech, sr = librosa.load(file_name, sr=sample_rate_khz, mono=True)
        os.remove(file_name)
        chunk_length = sr * chunk_duration
        overlap_length = sr * overlap_duration
        differential = chunk_length - overlap_length
        total_speech_length = len(speech)
        number_of_chunks = math.ceil(
            ((total_speech_length - chunk_length) / differential) + 1
        )
        for chunk_number in range(number_of_chunks):
            start_at_length = chunk_number * differential
            end_at_length = chunk_length + (chunk_number * differential)
            yield {
                "speech_block": speech[start_at_length:end_at_length],
                "start_time": start_at_length / sr,
                "end_time": end_at_length / sr,
            }

    def _chunk_to_text(self, chunk: np.array) -> str:
        input_values = self.tokenizer(chunk, return_tensors="pt").input_values.cuda()
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.decode(predicted_ids[0])
        return transcription

    def _transcribe_step(self, content_id: str, audio_url: str):
        if audio_url:
            content = requests.get(audio_url).content
            self._convert_mp3_to_wav(mp3_content=content, file_name=content_id)
            transcription = [
                (
                    self._chunk_to_text(chunk["speech_block"]),
                    chunk["start_time"],
                    chunk["end_time"],
                )
                for chunk in self._stream_chunks(
                    file_name=content_id,
                    sample_rate_khz=16000,
                    chunk_duration=20,
                    overlap_duration=1,
                )
                if any(chunk)
            ]
            return transcription

    def transcribe(self, podcast_data: List[pd.DataFrame]) -> dict:
        df_podcast = pd.concat(podcast_data)
        transcriptions = {
            content_id: self._transcribe_step(content_id, audio_url)
            for content_id, audio_url in tqdm(
                zip(df_podcast["content_id"], df_podcast["audio_url"])
            )
        }
        return transcriptions

    @staticmethod
    def get_podcast_data() -> Iterator[pd.DataFrame]:
        language = "en"
        page_number = 0
        data = list()
        while data or not page_number:
            podcast_response = requests.get(
                f"https://preprod.sadhguru.org/isha/api/podcasts?langcode={language}&page={page_number}&category=all"
            )
            data = podcast_response.json()["data"]
            page_number += 1
            yield pd.DataFrame(data)

    @staticmethod
    def to_dataset(
        podcast_data: List[pd.DataFrame], transcriptions: dict
    ) -> pd.DataFrame:
        assert (
            type(transcriptions) == dict
        ), f"Accepted type is [dict], passed type is [{type(transcriptions)}]"
        df_podcast = pd.concat(podcast_data)
        subtitles = {
            content_id: list(map(operator.itemgetter(0), value)) if value else [np.NaN]
            for content_id, value in transcriptions.items()
        }
        timestamps = {
            content_id: list(
                zip(
                    list(map(operator.itemgetter(1), value)),
                    list(map(operator.itemgetter(2), value)),
                )
            )
            if value
            else [np.NaN]
            for content_id, value in transcriptions.items()
        }
        df_podcast["subtitles"] = df_podcast["content_id"].map(subtitles)
        df_podcast["timestamps"] = df_podcast["content_id"].map(timestamps)
        df_podcast = df_podcast.rename(columns={"content_id": "videoId"})
        df_podcast = df_podcast.set_index("videoId")
        return df_podcast
