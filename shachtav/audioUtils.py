import librosa
import numpy as np

from . import config


def load_file(filename):
    audio, file_sr = librosa.load(filename)
    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=config.sample_rate)
    return audio


def calc_spectrogram(sound):
    spec = librosa.feature.melspectrogram(y=sound,
                                          sr=config.sample_rate,
                                          hop_length=config.hop_length,
                                          n_fft=config.n_fft,
                                          n_mels=config.spectrogram_n_bins,
                                          fmin=config.spectrogram_f_min,
                                          fmax=config.spectrogram_f_max)
    spec = librosa.power_to_db(spec, ref=np.max)

    spec = spec.T  # take transpose, so that shape is (time, bins) rather than (bins, time)
    spec = np.asarray(spec)

    return spec
