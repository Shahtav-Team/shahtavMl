import librosa
import music21

import sheet_music
from shachtav import WavToMidiModel
from shachtav.BeatDownbeatDetector import BeatDownbeatDetector
from shachtav.VoiceSplitting import ClefSplitModel
import numpy as np


def infer():
    wav_to_midi_model_path = "models/onsets_offsets_model"
    clef_split_model_path = "models/clefs_model_v1"
    song_path = "samples/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.wav"

    wav_to_midi = WavToMidiModel.WavToMidiModel.load(wav_to_midi_model_path)
    clef_split = ClefSplitModel.load(clef_split_model_path)
    beat_detector = BeatDownbeatDetector()

    audio, sr = librosa.load(song_path)
    audio = audio[:sr * 30]

    result = wav_to_midi.infer(audio, sr)
    song = result.decode()
    beat_info = beat_detector.find_beats_from_array(audio, sr, [3, 4])
    beat_info2 = beat_detector.find_beats(song_path, [3, 4])
    song, notes_split = clef_split.run(song, beat_info, 4)

    score = sheet_music.score_from_notes(notes_split, beat_info)
    filename = f"samples/test_{np.random.randint(0, 1000000)}.pdf"
    score.write("musicxml.pdf", filename)
    print(filename)


def test_musecore():
    score = music21.stream.Stream()
    score.append(music21.note.Note("C4"))
    score.write("musicxml.pdf", "samples/test.pdf")


if __name__ == "__main__":
    music21.environment.set('musescoreDirectPNGPath', "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe")
    infer()
