from dataclasses import dataclass
from typing import List

import keras
import numpy as np
import pretty_midi

from . import MidiEncoding
from . import config
from . import BeatDownbeatDetector
from .BeatDownbeatDetector import BeatInfo
import copy
import matplotlib.pyplot as plt


class Clef:
  bass = 0
  treble = 1
  no_note = 2

@dataclass
class NotesSplit:
    bass_notes: List[pretty_midi.Note]
    treble_notes: List[pretty_midi.Note]


def plot_clef_data(notes, clef):
  bass = np.nonzero((notes == 1) & (clef == Clef.bass))
  treble = np.nonzero((notes == 1) & (clef == Clef.treble))

  plt.figure(figsize = (15, 4))
  plt.xlim(0, 1000)
  plt.scatter(*treble, color = "green", s = 2)
  plt.scatter(*bass, color = "red", s = 2)
  plt.show()


def quantize_time(time, beat_times, resolution):
    i = np.searchsorted(beat_times, time, side="left")
    i = np.clip(i, 1,
                len(beat_times) - 1)  # clip i so that beat_times[i - 1] and beat_times[i] are not index out of bounds.
    beat_before = beat_times[i - 1]
    beat_after = beat_times[i]

    beat_fraction = (time - beat_before) / (beat_after - beat_before)

    beat_time = (i - 1) * resolution + round(resolution * beat_fraction)

    # make sure our time is within bounds of the beat.
    # todo: is this the best way to do this? What if a note lasts for after the end of the beat.
    beat_time = np.clip(beat_time, 0, resolution * len(beat_times))
    return beat_time

def quantize_midi(midi, beat_times, resolution):
    notes_quantized = []
    for note in midi.instruments[0].notes:
        note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=quantize_time(note.start, beat_times, resolution),
            end=quantize_time(note.end, beat_times, resolution)
        )

        notes_quantized.append(note)
    midi_quantized = MidiEncoding.notes_to_pretty_midi(notes_quantized, bpm=60 / resolution)
    return midi_quantized

def quantized_midi_to_onsets(midi, length):
    onsets = np.zeros((length, config.midi_num_pitches), dtype=np.int32)
    for note in midi.instruments[0].notes:
        pitch_norm = note.pitch - config.midi_pitch_min
        start_t = int(note.start)
        onsets[start_t, pitch_norm] = 1
    return onsets


class ClefSplitModel:
    @staticmethod
    def load(model_path):
        model = keras.models.load_model(model_path, compile=False)
        return ClefSplitModel(model)

    def __init__(self, model):
        self.model = model

    def split_clefs(self, quantized_mid):
        onsets = quantized_midi_to_onsets(quantized_mid, int(quantized_mid.get_end_time()) + 1)
        onsets_expanded = np.expand_dims(onsets, axis=0)  # add extra dimension to array so that it can be passed to the model.
        clefs = self.model(onsets_expanded)
        clefs = np.asarray(clefs)
        clefs = clefs.squeeze()
        clefs = clefs > 0

        bass_notes = []
        treble_notes = []

        for note in quantized_mid.instruments[0].notes:
            pitch_norm = note.pitch - config.midi_pitch_min
            start_t = int(note.start)

            if clefs[start_t, pitch_norm] == Clef.treble:
                treble_notes.append(note)
            else:
                bass_notes.append(note)

        return bass_notes, treble_notes, onsets, clefs

    def run(self, song: MidiEncoding.Song, beat_info: BeatInfo, quantize_resolution: int = 4):
        """
        Identifies which notes should be put in the top of the grand staff (voice = 1) and the bottom of the grand staff (voice = 0)
        Args:
            song: the output of WavToMidiModel
            beat_info: the output of BeatDownbeatDetector
            quantize_resolution: how many units of time should be in a signle beat after quantization.
        Returns:
            song: the input song, modified to include voices.
            notesSplit: NotesSplit. a NotesSplit object which determines which notes are on which staff.
            or None if the song was too short and the output can't be determined.

        """
        song = copy.deepcopy(song)
        if len(beat_info.beats) <= 1:
            return song, None
        midi = song.to_pretty_midi()
        midi_quantized = quantize_midi(midi, beat_info.beats, quantize_resolution)
        bass_notes, treble_notes, onsets, clefs = self.split_clefs(midi_quantized)

        for note in song.notes:
            start_secs = note.start * song.frame_length_seconds
            start_index = quantize_time(start_secs, beat_info.beats, resolution=quantize_resolution)
            pitch_norm = note.pitch - config.midi_pitch_min
            voice = int(clefs[start_index, pitch_norm])
            note.voice = voice
        return song, NotesSplit(bass_notes, treble_notes)