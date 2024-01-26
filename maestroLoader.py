import os
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


def load_song(midi_file, audio_file):
    audio = audioUtils.load_file(audio_file)
    spectrogram = audioUtils.calc_spectrogram(audio)

    midi = pretty_midi.PrettyMIDI(midi_file)
    midi_encoding = MidiEncoding.from_pretty_midi(midi, config.frame_length_seconds,
                                                  onset_length_frames=config.encoding_onset_length_frames,
                                                  offset_length_frames=config.encoding_offset_length_frames)

    # crop the spectrogram and midi encoding to be the same length
    spectrogram = spectrogram[: midi_encoding.length_frames()]
    midi_encoding.cop_to_length(len(spectrogram))
    assert len(spectrogram) == midi_encoding.length_frames()

    return dict(
        spectrogram=spectrogram,
        **midi_encoding.to_dict()
    )


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
        yield load_song(song["midi_filename"], song["audio_filename"])


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
