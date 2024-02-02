import librosa
import music21
import numpy as np

import sheet_music
from shachtav import WavToMidiModel
from shachtav.BeatDownbeatDetector import BeatDownbeatDetector
from shachtav.VoiceSplitting import ClefSplitModel

import sample_song



def infer():
    wav_to_midi_model_path = "models/onsets_offsets_model"
    clef_split_model_path = "models/clefs_model_v1"
    song_path = "samples/turkish_march.mp3"

    # wav_to_midi = WavToMidiModel.WavToMidiModel.load(wav_to_midi_model_path)
    clef_split = ClefSplitModel.load(clef_split_model_path)
    beat_detector = BeatDownbeatDetector()

    audio, sr = librosa.load(song_path)
    # audio = audio[:sr * 30]

    # result = wav_to_midi.infer(audio, sr)
    song = sample_song.song

    beat_info = beat_detector.find_beats_from_array(audio, sr, [3, 4])
    song, notes_split = clef_split.run(song, beat_info, 4)

    score = sheet_music.score_from_notes(notes_split, beat_info)
    filename = f"samples/test_{np.random.randint(0, 1000000)}.pdf"
    score.write("musicxml.pdf", filename)
    print(filename)


def test_musecore():
    score = music21.stream.Stream()
    score.insert(0, music21.key.KeySignature(2))
    score.insert(2, music21.note.Note("C4"))
    score.insert(3, music21.note.Note("D4"))
    score.insert(4, music21.note.Note("F4", quarterLength=2))
    score.insert(6, music21.note.Note("E4", quarterLength=3))
    score.insert(8, music21.note.Note("F4", quarterLength=2))

    score.makeMeasures(inPlace=True)

    score.shiftElements(-2, startOffset=1 / 8)
    mes1 = score.measure(1)
    mes1.shiftElements(-2, startOffset=1 / 8)
    mes1.paddingLeft = 2
    print(mes1.barDuration.quarterLength - mes1.paddingLeft)

    for measure in score.getElementsByClass(music21.stream.Measure):
        measure_length = measure.barDuration.quarterLength - measure.paddingLeft
        for note in measure.notes:
            if note.offset + note.quarterLength > measure_length:
                print()
                print()
                print(f"Shortening note: {note}")
                print(f"Offset: {note.offset}")
                print(f"Q Len: {note.quarterLength}")
                print(f"mes len: {measure_length}")
                print()
                print()
                note.quarterLength = measure_length - note.offset

    score.makeRests(inPlace=True, timeRangeFromBarDuration=True)
    score.show("text")


if __name__ == "__main__":
    music21.environment.set('musescoreDirectPNGPath', "C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe")
    infer()
