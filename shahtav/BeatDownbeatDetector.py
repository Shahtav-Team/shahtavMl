# Workaround for using madmom with python 3.10 or later, because madmom uses deprecated
import collections

import librosa
import numpy as np

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence

from dataclasses import dataclass

import madmom

int_16_max = 32767
madmom_sr = 44100

@dataclass
class BeatInfo:
    beats: np.ndarray
    is_downbeat: np.ndarray
    beats_per_bar: int


class BeatDownbeatDetector:
    def __init__(self):
        self.rnn_processor = madmom.features.RNNDownBeatProcessor(num_threads=4)

    def find_beats(self, filename, allowed_beats_per_bar):
        # todo make this work with an already loaded file
        # should normalize audio to 16 bit so it works with library.
        dbn_processor = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=allowed_beats_per_bar, fps=100)
        rnn_result = self.rnn_processor(filename)
        beats_estimate = dbn_processor(rnn_result)
        is_downbeat = beats_estimate[:, 1] == 1
        downbeats = beats_estimate[is_downbeat][:, 0]
        beats = beats_estimate[:, 0]
        beats_per_bar = np.max(beats_estimate[:, 1])
        return BeatInfo(
            beats=beats,
            is_downbeat=is_downbeat,
            beats_per_bar=beats_per_bar
        )

    def find_beats_from_array(self, array, sample_rate, allowed_beats_per_bar):
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=madmom_sr)
        dbn_processor = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=allowed_beats_per_bar, fps=100)
        array_scaled = int_16_max * array
        array_scaled = array_scaled.astype(np.int16)
        rnn_result = self.rnn_processor.process(array_scaled)
        beats_estimate = dbn_processor(rnn_result)
        is_downbeat = beats_estimate[:, 1] == 1
        downbeats = beats_estimate[is_downbeat][:, 0]
        beats = beats_estimate[:, 0]
        if len(beats_estimate) != 0:
            beats_per_bar = np.max(beats_estimate[:, 1])
        else:
            beats_per_bar = 4
        return BeatInfo(
            beats=beats,
            is_downbeat=is_downbeat,
            beats_per_bar=beats_per_bar
        )
