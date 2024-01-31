import os
import random

import audiomentations

import librosa
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm
import tensorflow as tf

import audioUtils
import config
import utils
from MidiEncoding import MidiEncoding
from utils import split_array_into_chunks

sparse_keys = ["frames", "onsets", "offsets", "velocities"]


def load_or_cache(songs_paths, cache_path):
    if not os.path.exists(cache_path):
        print(f"Creating Dataset cache {cache_path}")
        dataset = load_songs_dataset(songs_paths)\
            .apply(split_dataset_to_chunks)\
            .map(lambda x: utils.dense_to_spare(x, sparse_keys))
        dataset.save(cache_path)
    return tf.data.Dataset.load(cache_path)


def load_maestro_train(maestro_path, cache_path, max_songs = None):
    train_filenames, _, _ = load_maestro_filenames(maestro_path)
    if max_songs is not None:
        train_filenames = train_filenames[:max_songs]
    cache = os.path.join(cache_path, "train_cache.tfrecord")
    return load_or_cache(train_filenames, cache)


def load_maestro_valid(maestro_path, cache_path, max_songs = None):
    _, _, valid_filenames = load_maestro_filenames(maestro_path)
    if max_songs is not None:
        valid_filenames = valid_filenames[:max_songs]
    cache = os.path.join(cache_path, "valid_cache.tfrecord")
    return load_or_cache(valid_filenames, cache)


def load_maestro_test(maestro_path, cache_path, max_songs = None):
    _, test_filenames, _ = load_maestro_filenames(maestro_path)
    if max_songs is not None:
        test_filenames = test_filenames[:max_songs]
    cache = os.path.join(cache_path, "test_cache.tfrecord")
    return load_or_cache(test_filenames, cache)


def load_song(midi_file, audio_file, noise=False):
    audio = audioUtils.load_file(audio_file)

    if noise:
        audio = add_noise(audio, config.noise_path)

    spectrogram = audioUtils.calc_spectrogram(audio)

    midi = pretty_midi.PrettyMIDI(midi_file)
    midi_encoding = MidiEncoding.from_pretty_midi(midi, config.frame_length_seconds)

    # crop the empty end of the spectrogram to make it the same shape as the midi
    spectrogram = spectrogram[: midi_encoding.length_frames()]
    assert len(spectrogram) == midi_encoding.length_frames()

    return dict(
        spectrogram=spectrogram,
        **midi_encoding.to_dict()
    )


def add_noise(audio, noise_path):
    background_noise = audiomentations.AddBackgroundNoise(
        sounds_path=noise_path,
        min_snr_db=19.0,
        max_snr_db=25.0,
        p=0.7
    )

    air_absorption = audiomentations.AirAbsorption(
        p=1
    )

    gausian_noise = audiomentations.AddGaussianSNR(
        min_snr_db=45.0,
        max_snr_db=55.0,
        p=1
    )

    return gausian_noise(air_absorption(
            background_noise(audio, config.sample_rate), config.sample_rate), config.sample_rate)


def add_noise_legacy(audio, noise_path):
    # get the list of noise file
    noise_file = random.choice(os.listdir(noise_path))

    noise_audio = audioUtils.load_file(os.path.join(noise_path, noise_file))

    # normalize the volume
    noise_audio = librosa.util.normalize(noise_audio)

    noise_audio = np.tile(noise_audio, 1 + (audio.size // noise_audio.size))
    noise_audio = noise_audio[:audio.size]

    return audio * (1 - config.noise_percentage) + noise_audio * config.noise_percentage


def split_dataset_to_chunks(songs_dataset):
    songs_ds = tf.data.Dataset.from_generator(
        lambda: chunk_songs_lazy(songs_dataset),
        output_signature={
            "spectrogram": tf.TensorSpec(shape=(config.chunk_length_frames, config.spectrogram_n_bins),
                                         dtype=tf.float32, name="spectrogram"),
            "frames": tf.TensorSpec(shape=(config.chunk_length_frames, config.midi_num_pitches), dtype=tf.float32,
                                    name="frames"),
            "onsets": tf.TensorSpec(shape=(config.chunk_length_frames, config.midi_num_pitches), dtype=tf.float32,
                                    name="onsets"),
            "offsets": tf.TensorSpec(shape=(config.chunk_length_frames, config.midi_num_pitches), dtype=tf.float32,
                                     name="offsets"),
            "velocities": tf.TensorSpec(shape=(config.chunk_length_frames, config.midi_num_pitches), dtype=tf.float32,
                                        name="velocities")
        }
    )
    return songs_ds


def chunk_songs_lazy(songs_dataset):
    for song in songs_dataset:
        spectrogram_split = split_array_into_chunks(np.asarray(song["spectrogram"]), config.chunk_length_frames,
                                                    pad_value=-80)
        frames_split = split_array_into_chunks(np.asarray(song["frames"]), config.chunk_length_frames)
        onsets_split = split_array_into_chunks(np.asarray(song["onsets"]), config.chunk_length_frames)
        offsets_split = split_array_into_chunks(np.asarray(song["offsets"]), config.chunk_length_frames)
        velocities_split = split_array_into_chunks(np.asarray(song["velocities"]), config.chunk_length_frames)

        for spectrogram, frames, onsets, offsets, velocities \
                in zip(spectrogram_split, frames_split, onsets_split, offsets_split, velocities_split):
            yield {
                "spectrogram": spectrogram,
                "frames": frames,
                "onsets": onsets,
                "offsets": offsets,
                "velocities": velocities,
            }


def load_songs_lazy(songs_paths):
    iterable = list(songs_paths.iterrows())
    iterable = tqdm(iterable)
    for index, song in iterable:
        yield load_song(song["midi_filename"], song["audio_filename"], True)


def load_songs_dataset(songs_paths):
    songs_ds = tf.data.Dataset.from_generator(
        lambda: load_songs_lazy(songs_paths),
        output_signature={
            "spectrogram": tf.TensorSpec(shape=(None, config.spectrogram_n_bins), dtype=tf.float32, name="spectrogram"),
            "frames": tf.TensorSpec(shape=(None, config.midi_num_pitches), dtype=tf.float32, name="frames"),
            "onsets": tf.TensorSpec(shape=(None, config.midi_num_pitches), dtype=tf.float32, name="onsets"),
            "offsets": tf.TensorSpec(shape=(None, config.midi_num_pitches), dtype=tf.float32, name="offsets"),
            "velocities": tf.TensorSpec(shape=(None, config.midi_num_pitches), dtype=tf.float32, name="velocities")
        }
    )
    return songs_ds


def load_maestro_filenames(maestro_path):
    filenames_csv = os.path.join(maestro_path, "maestro-v3.0.0.csv")
    filenames_df = pd.read_csv(filenames_csv)
    # make the filenames in the dataset absolute instead of relative to maestro folder.
    filenames_df["midi_filename"] = filenames_df["midi_filename"].apply(
        lambda filename: os.path.join(maestro_path, filename)
    )
    filenames_df["audio_filename"] = filenames_df["audio_filename"].apply(
        lambda filename: os.path.join(maestro_path, filename)
    )
    train_filenames = filenames_df[filenames_df["split"] == "train"]
    test_filenames = filenames_df[filenames_df["split"] == "test"]
    valid_filenames = filenames_df[filenames_df["split"] == "validation"]
    # ensure each example is put into test, train, or valid
    assert len(train_filenames.index) + len(test_filenames.index) + len(valid_filenames.index) == len(
        filenames_df.index)
    return train_filenames, test_filenames, valid_filenames
