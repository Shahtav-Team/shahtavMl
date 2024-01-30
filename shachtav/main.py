import pretty_midi
import config
import visual_midi

from .WavToMidiModel import WavToMidiModel
from .MidiEncoding import MidiEncoding
import librosa
import keras
def infer():
    model_path = "models/onsets_offset_modelv1"
    song_path = "samples/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav"

    model = WavToMidiModel.load(model_path)
    audio, sr = librosa.load(song_path)
    audio = audio[:sr * 60]
    result = model.infer(audio, sr)

    midi = result.decode().to_pretty_midi()
    midi.write("samples/test.mid")

def plot_model():
    model = WavToMidiModel.create()

    model.model.summary()
    print()
    print()

    model.model.get_layer("onsets_acoustic_model").summary()


if __name__ == "__main__":
    plot_model()
