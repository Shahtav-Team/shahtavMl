import keras
import tensorflow as tf
import librosa
import numpy as np
from keras import layers
from dataclasses import dataclass

import audioUtils
import config
import maestroLoader
import utils
from MidiEncoding import MidiEncoding


@keras.saving.register_keras_serializable()
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)

    y_pred_masked = tf.boolean_mask(y_pred, mask)
    y_true_masked = tf.boolean_mask(y_true, mask)
    # in the rare case where the targets are empty, return 0 instead of nan
    if len(y_true_masked) == 0:
        return tf.constant(0)
    # When working with binary_crossentropy for regression on targets that are not 0 or 1,
    # make the loss relative to the lowest possible loss given the target, so that a perfect model has a loss of 0.
    crossentropy = keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)
    crossentropy_min = keras.losses.binary_crossentropy(y_true_masked, y_true_masked)

    return crossentropy - crossentropy_min

class WavToMidiModel:
    @dataclass
    class Params:
        base_lr: int = 0.0006
        lr_decay: int = 0.93
        lr_decay_steps: int = 10000
        batch_size = 16

    @staticmethod
    def _acoustic_model(inp_shape, name):
        model = keras.Sequential(name=name)
        model.add(keras.Input(shape=inp_shape))
        model.add(layers.Reshape((-1, config.spectrogram_n_bins, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=48, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=48, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling2D(pool_size=(1, 2)))
        model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.MaxPooling2D(pool_size=(1, 2)))
        # flatten along frequency and channel axis
        _, times, frequencies, channels = model.get_layer(index=-1).output.shape
        model.add(layers.Reshape((-1, frequencies * channels)))
        model.add(layers.Conv1D(filters=768, kernel_size=1, activation="relu"))
        model.add(layers.Dropout(0.5))

        return model

    @staticmethod
    def create(params: Params = None):
        if params is None:
            params = WavToMidiModel.Params()

        input_shape = (None, config.spectrogram_n_bins)
        output_shape = (None, config.midi_num_pitches)

        inputs = keras.Input(input_shape, name="spectrogram")

        onsets_acoustic = WavToMidiModel._acoustic_model(input_shape, "onsets_acoustic_model")(inputs)
        onsets_memory = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name="onsets_memory")(
            onsets_acoustic)
        onsets_pred = layers.Conv1D(filters=config.midi_num_pitches, kernel_size=1, activation="sigmoid",
                                    name="onsets_out")(onsets_memory)

        offsets_acoustic = WavToMidiModel._acoustic_model(input_shape, "offsets_acoustic_model")(inputs)
        offsets_memory = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name="offsets_memory")(
            offsets_acoustic)
        offsets_pred = layers.Conv1D(filters=config.midi_num_pitches, kernel_size=1, activation="sigmoid",
                                     name="offsets_out")(offsets_memory)

        frames_acoustic = WavToMidiModel._acoustic_model(input_shape, "frames_acoustic_model")(inputs)

        combined = layers.Concatenate(name="combine")([frames_acoustic, onsets_pred, offsets_pred])
        combined_memory = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name="combined_memory")(
            combined)
        frames_pred = layers.Conv1D(filters=config.midi_num_pitches, kernel_size=1, activation="sigmoid",
                                    name="frames_out")(
            combined_memory)

        velocities_acoustic = WavToMidiModel._acoustic_model(input_shape, "velocities_acoustic_model")(inputs)
        velocities_memory = layers.Bidirectional(layers.LSTM(256, return_sequences=True), name="velocities_memory")(
            velocities_acoustic)
        velocities_pred = layers.Conv1D(filters=config.midi_num_pitches, kernel_size=1, activation="sigmoid",
                                       name="velocities_out")(velocities_memory)


        model = keras.Model(
            inputs={
                "spectrogram": inputs
            },
            outputs={
                "onsets": onsets_pred,
                "offsets": offsets_pred,
                "frames": frames_pred,
                "velocities": velocities_pred
            }
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=params.base_lr
            ),
            loss={
                "onsets": keras.losses.BinaryCrossentropy(),
                "offsets": keras.losses.BinaryCrossentropy(),
                "frames": keras.losses.BinaryCrossentropy(),
                "velocities": masked_binary_crossentropy
            }
        )

        return WavToMidiModel(model, params)

    @staticmethod
    def load(filename, params=None):
        if params is None:
            params = WavToMidiModel.Params()
        model = keras.models.load_model(filename)
        return WavToMidiModel(model, params)

    def __init__(self, model, params):
        self.model = model
        self.params = params

    def train(self,
              train_data,
              valid_data,
              num_steps,
              checkpoint_path):
        """
        trains the model.
        train_data: a tf.Dataset object, containing dictionaries of the following types:
            - frames
            - onsets
            - offsets
            - spectrogram
        valid_data:
            - dataset with same shape as train data, but for validation
        num_steps: how many steps of training to run.
        checkpoint_path: path to save model checkpoints to.
        """

        model = self.model

        # map dataset so it works as needed for train API
        train_dataset = train_data \
            .cache() \
            .shuffle(10000) \
            .map(lambda x: utils.sparse_to_dense(x, maestroLoader.sparse_keys)) \
            .map(lambda x:
                 (
                     {"spectrogram": x["spectrogram"]},
                     {
                         "frames": x["frames"],
                         "onsets": x["onsets"],
                         "offsets": x["offsets"],
                         "velocities": x["velocities"]
                     }
                 )
                 ) \
            .batch(self.params.batch_size) \
            .prefetch(2)

        train_dataset_epoch_steps = train_dataset.cardinality().numpy()

        # load all the valid dataset into an array, so it works with the train API
        valid_data = valid_data \
            .map(lambda x: utils.sparse_to_dense(x, maestroLoader.sparse_keys)) \
            .map(lambda x:
                 (
                     {"spectrogram": x["spectrogram"]},
                     {
                         "frames": x["frames"],
                         "onsets": x["onsets"],
                         "offsets": x["offsets"],
                         "velocities": x["velocities"]
                     }
                 )
                 ) \
            .batch(valid_data.cardinality()) \
            .get_single_element()

        def lr_scheduler(epoch, lr):
            return self.params.base_lr * self.params.lr_decay ** int(
                epoch * train_dataset_epoch_steps / self.params.lr_decay_steps)

        lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=False)

        history = model.fit(train_dataset,
                            epochs=int(num_steps // train_dataset_epoch_steps) + 1,
                            callbacks=[lr_callback, model_checkpoint_callback],
                            validation_data=valid_data
                            )
        return history

    def infer(self, audio, sr) -> MidiEncoding:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate)
        spectrogram = audioUtils.calc_spectrogram(audio)
        # run with a batch size of 1, so we need to add an extra dimension to our array.
        spectrogram = np.expand_dims(spectrogram, 0)
        result = self.model(spectrogram)
        result["onsets"] = result["onsets"].numpy().squeeze()
        result["offsets"] = result["offsets"].numpy().squeeze()
        result["frames"] = result["frames"].numpy().squeeze()
        result["velocities"] = result["velocities"].numpy().squeeze()

        return MidiEncoding.from_dict(result, config.frame_length_seconds)
