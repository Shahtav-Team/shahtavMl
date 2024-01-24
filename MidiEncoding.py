import copy
import itertools
from typing import List
import pretty_midi
from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI
import numpy as np
from dataclasses import dataclass
import librosa

import config

SUSTAIN_NO = 64


def notes_to_pretty_midi(notes: List[pretty_midi.Note], bpm=120) -> PrettyMIDI:
    """
    utility function for creating a pretty midi object from a list of pretty midi notes.
    :param notes: a list of pretty notes.
    :param bpm: the tempo of the returned prettyMIDI object, in beats per minute. Sometimes imparts how the pretty midi object is displayed.
    :return: a pretty midi object, with a single instrument containing just these notes.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(0)
    instrument.notes = notes
    midi.instruments.append(instrument)
    return midi


def note_data_to_points(note_data):
    """
    For visualizing the midi representation. Converts from a list of points where the note_data is active.
    :param note_data: a 2d array of shape(time, midi_num_pitches)
    :return: a list of coordinates on the spectogram where the note_data is not zero.
    """
    note_points = [[], []]
    for frame_num in range(len(note_data)):
        time_sec = frame_num * config.frame_length_seconds
        active_midi = np.nonzero(note_data[frame_num])[0] + config.midi_pitch_min
        active_hz = librosa.midi_to_hz(active_midi)
        for freq in active_hz:
            note_points[0].append(time_sec)
            note_points[1].append(freq)

    note_points = np.array(note_points)
    return note_points


@dataclass
class Thresholds:
    """
    The output of the network is an array of probabilities for MIDI events, and we need to decode those events into actual evnets.
    This class denotes the thresholds for when the network output is considered a real midi event.
    """
    onset_threshold: float = 0.5
    offset_threshold: float = 0.5
    frame_threshold: float = 0.5


@dataclass
class MidiEncoding:
    """
    A representation of midi in a way that can efficiently be learned by a neural network. Midi is represented by 4 arrays. Of shape (num_frames, num_pitches)
     The first dimension of each of the arrays is time each index representing 'chunk_length_frames' seconds.
    The second dimension is the pitch of the notes in midi, normalized to only contain pitches present in a piano. pitch_norm = pitch_midi - midi_pitch_min.
    The arrays are:
    onsets: this array is 0 everywhere except for wherever a notes starts in which case it has value 1.
    offsets: this array is 0 everywhere except for at the last frame of a note, where it has value 1.
    frames: this array is 1 wherever the note is playing, or 0 wherever it isn't.
    velocities: on note onset, this value is equal to the midi velocity, normalized to the range (0,1]. Everywhere else, this value is unspecified.
     If loaded directly from mid, it should be set to 0. If in the output of the network, it may be anything the network decides to output.
    TODO: add velocity.
    """

    onsets: np.ndarray
    offsets: np.ndarray
    frames: np.ndarray
    velocities: np.ndarray
    frame_length_seconds: float

    @staticmethod
    def from_pretty_midi(midi: PrettyMIDI, frame_length_seconds: float,
                         onset_length_frames=1, offset_length_frames=1):
        """
        Encodes a midi object in this format.
        midi: a PrettyMIDI object which we want to encode. Should only have 1 instrument which contains all the notes we want to encode.
        frame_length_seconds: how many seconds a single frame of the time axis represents.
        onset_frame_length: how many frames should be marked as an onset during an onset. Having a larger value helps the network learn if onset data is imprecise.
        offset_frame_length: how many frames should be marked as an offset during an offset.
        """

        array_length = round(midi.get_end_time() / frame_length_seconds) + 1
        frames = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        onsets = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        offsets = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        velocities = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)

        midi = MidiEncoding.extend_with_sustain(midi)

        for note in midi.instruments[0].notes:
            pitch = note.pitch
            start = round(note.start / frame_length_seconds)
            end = round(note.end / frame_length_seconds)
            pitch_norm = pitch - config.midi_pitch_min
            frames[start:end, pitch_norm] = 1
            onsets[start: start + onset_length_frames, pitch_norm] = 1
            velocities[start:start + onset_length_frames, pitch_norm] = note.velocity / 127
            offsets[end: end + offset_length_frames, pitch_norm] = 1
        return MidiEncoding(
            onsets=onsets,
            offsets=offsets,
            frames=frames,
            velocities=velocities,
            frame_length_seconds=frame_length_seconds
        )

    @staticmethod
    def get_sustain_events(midi: PrettyMIDI):
        """
        Return a list of sustain events
        midi: a PrettyMidi object which we want to get the sustain event from
        returns a list of objects containing start and end times
        """
        sustain_events = []

        current_sustain = None

        for control_change in sorted(midi.instruments[0].control_changes, key=lambda x: x.time):
            if control_change.number == SUSTAIN_NO:
                if control_change.value >= 64 and current_sustain is None:
                    current_sustain = {"start": control_change.time}
                elif control_change.value < 64 and current_sustain is not None:
                    current_sustain["end"] = control_change.time
                    sustain_events.append(current_sustain)
                    current_sustain = None

        if current_sustain is not None:
            current_sustain["end"] = midi.get_end_time()
            sustain_events.append(current_sustain)

        return sustain_events

    @staticmethod
    def extend_with_sustain(midi: PrettyMIDI):
        """
        Gets a pretty midi object
        Args:
            midi: A PrettyMIDI object, that we want to expand the notes of

        Returns:
            A modified version of the PrettyMIDI object, where notes are expanded
            using the sustain pedal
        """
        midi_copy = copy.deepcopy(midi)

        # create a bin for each possible pitch
        note_bins = [[] for _ in range(128)]

        # add the notes to the correct bins
        for note in midi_copy.instruments[0].notes:
            note_bins[note.pitch].append(note)

        # get the sustain events
        sustain_events = MidiEncoding.get_sustain_events(midi_copy)

        # lengthen the notes based on the sustain
        for note_bin in note_bins:
            # find the last sustain event played during the note
            for i, note in enumerate(note_bin):
                sustain_end = list(filter(
                    lambda x: x["start"] <= note.end <= x["end"],
                    sustain_events)
                    )
                if len(sustain_end) > 0:
                    note.end = sustain_end[0]["end"]

                # check a new note starts before the end
                if i + 1 < len(note_bin) and note.end >= note_bin[i + 1].start:
                    note.end = note_bin[i].start
        midi_copy.instruments[0].notes = list(sorted(
            itertools.chain.from_iterable(note_bins),
            key=lambda x: x.time))

        midi_copy.instruments[0].control_changes = []

        return midi_copy


    def to_pretty_midi(self, thresholds: Thresholds = None) -> pretty_midi.PrettyMIDI:
        if thresholds is None:
            thresholds = Thresholds()
        onsets = self.onsets > thresholds.onset_threshold
        offsets = self.offsets > thresholds.offset_threshold
        frames = self.frames > thresholds.frame_threshold

        notes = []
        for pitch_id in range(config.midi_num_pitches):
            pitch_midi = pitch_id + config.midi_pitch_min

            onset_frame_nums, = np.nonzero(onsets[:, pitch_id])
            curr_frame = 0
            for onset_frame in onset_frame_nums:
                if curr_frame > onset_frame:
                    continue
                curr_frame = onset_frame

                # the note ends when {offsets} is true, or when {frames} is not true
                while curr_frame < offsets.shape[0]:
                    is_onset = onsets[curr_frame, pitch_id]
                    is_offset = offsets[curr_frame, pitch_id]
                    is_frame = frames[curr_frame, pitch_id]

                    if not is_onset and (is_offset or not is_frame):
                        # the note has ended
                        break
                    curr_frame += 1
                offset_frame = curr_frame

                onset_time = onset_frame * self.frame_length_seconds
                offset_time = offset_frame * self.frame_length_seconds
                note = pretty_midi.Note(velocity=64,
                                        pitch=pitch_midi,
                                        start=onset_time,
                                        end=offset_time)
                notes.append(note)
        notes.sort(key=lambda note: note.start)
        return notes_to_pretty_midi(notes)

    def length_frames(self):
        return self.frames.shape[0]

    @staticmethod
    def from_dict(dict, frame_length_seconds):
        """
        Creates an instance of this class from a dict, which is returned by the Keras model API.
        Also needs to be proved the frame length used.
        """
        return MidiEncoding(
            onsets=np.asarray(dict["onsets"]),
            offsets=np.asarray(dict["offsets"]),
            frames=np.asarray(dict["frames"]),
            velocities=np.asarray(dict["velocities"]),
            frame_length_seconds=frame_length_seconds
        )

    """
    Returns the representation of this encoding as a dictionary, which is needed for the keras training API.
    """

    def to_dict(self):
        return {
            "onsets": self.onsets,
            "offsets": self.offsets,
            "frames": self.frames,
            "velocities": self.velocities,
        }

    def plot_on_spectrogram(self, spectrogram):
        if spectrogram is not None:
            spectrogram = np.asarray(spectrogram)
            spectrogram = spectrogram.T  # take the transpose of the spectrogram so it's plotted correctly
            librosa.display.specshow(spectrogram, sr=config.sample_rate, hop_length=config.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
        frame_points = note_data_to_points(self.frames)
        plt.scatter(frame_points[0, :], frame_points[1, :], marker=",", s=3, alpha=0.5, color="blue")
        onset_points = note_data_to_points(self.onsets)
        plt.scatter(onset_points[0], onset_points[1], marker=",", s=3, alpha=0.5, color="green")
        offset_points = note_data_to_points(self.offsets)
        plt.scatter(offset_points[0, :], offset_points[1, :], marker=",", s=3, alpha=0.5, color="red")

        plt.show()
