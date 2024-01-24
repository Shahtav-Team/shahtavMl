import audioUtils
import maestroLoader
import soundfile as sf
import config

from WavToMidiModel import WavToMidiModel
from MidiEncoding import MidiEncoding
import librosa
def infer():
    model_path = "models/onsets_offset_modelv1"
    song_path = "samples/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav"

    model = WavToMidiModel.load(model_path)
    audio, sr = librosa.load(song_path)
    audio = audio[:sr * 60]
    result = model.infer(audio, sr)

    midi = result.to_pretty_midi()
    midi.write("samples/test.mid")


if __name__ == "__main__":
    audio = audioUtils.load_file("./samples/song.wav")
    audio = maestroLoader.add_noise(audio, config.noise_path)

    sf.write('./output.wav', audio, config.sample_rate, subtype='PCM_24')