import copy
import itertools
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np
import pretty_midi
from matplotlib import pyplot as plt
from pretty_midi import PrettyMIDI

from shachtav import config

SUSTAIN_NO = 64


def keep_first_true(arr):
    """
    Given a numpy array of booleans, returns the array but with all sequences of the value True
    turned into False except for the first element of the sequence using vectorization.

    Args:
        arr: A numpy array of booleans.

    Returns:
        A numpy array of booleans with the specified modification.
    """
    roll_mask = np.roll(arr, 1)
    roll_mask[0] = False
    return arr & ~roll_mask


def plot_midi(midi, plot_height=650, show_beat=False):
    import visual_midi
    preset = visual_midi.Preset(plot_height=plot_height, show_bar=False, show_beat=show_beat)
    plotter = visual_midi.Plotter(preset)
    plot = plotter.show(midi, f"out_{np.random.randint(0, 1000000)}.html")


def notes_to_pretty_midi(notes: List[pretty_midi.Note], bpm=120, control_changes=None) -> PrettyMIDI:
    """
    utility function for creating a pretty midi object from a list of pretty midi notes.
    :param notes: a list of pretty notes.
    :param bpm: the tempo of the returned prettyMIDI object, in beats per minute. Sometimes imparts how the pretty midi object is displayed.
    :return: a pretty midi object, with a single instrument containing just these notes.
    :param control_changes: list of pretty midi control changes, or None if there are no control changes
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(0)
    instrument.notes = notes
    if control_changes is not None:
        instrument.control_changes = control_changes
    midi.instruments.append(instrument)
    return midi


def note_data_to_points(note_data):
    """
    For visualizing the midi representation. Converts from a list of points where the note_data is active.
    :param note_data: a 2d array of shape(time, midi_num_pitches)
    :return: a list of coordinates on the spectrogram where the note_data is not zero.
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
    The output of the network is an array of probabilities for MIDI events, and we need to decode those events into actual events.
    This class denotes the thresholds for when the network output is considered a real midi event.
    """
    onset_threshold: float = 0.5
    offset_threshold: float = 0.5
    frame_threshold: float = 0.5

    pedal_on_threshold = 0.7
    pedal_off_threshold = 0.3

    min_note_length_frames = 1


@dataclass
class Note:
    start: int
    length: int
    pitch: int
    velocity: int
    voice: int = 0


@dataclass
class PedalEvent:
    time: int
    on: bool


@dataclass
class Song:
    frame_length_seconds: float
    notes: list[Note]
    pedal_events: list[PedalEvent]

    def to_pretty_midi(self):
        pedal_events = []
        notes = []

        for event in self.pedal_events:
            pedal_events.append(
                pretty_midi.ControlChange(number=SUSTAIN_NO,
                                          value=127 if event.on else 0,
                                          time=event.time * self.frame_length_seconds)
            )

        for note in self.notes:
            notes.append(
                pretty_midi.Note(velocity=note.velocity,
                                 pitch=note.pitch,
                                 start=note.start * self.frame_length_seconds,
                                 end=(note.start + note.length) * self.frame_length_seconds)
            )

        return notes_to_pretty_midi(notes, control_changes=pedal_events)


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
    """

    onsets: np.ndarray
    offsets: np.ndarray
    frames: np.ndarray
    velocities: np.ndarray
    pedals: np.ndarray
    frame_length_seconds: float

    @staticmethod
    def from_pretty_midi(midi: PrettyMIDI, frame_length_seconds: float,
                         onset_length_frames=1, offset_length_frames=1,
                         array_length=None):
        """
        Encodes a midi object in this format.
        midi: a PrettyMIDI object which we want to encode. Should only have 1 instrument which contains all the notes we want to encode.
        frame_length_seconds: how many seconds a single frame of the time axis represents.
        onset_frame_length: how many frames should be marked as an onset during an onset. Having a larger value helps the network learn if onset data is imprecise.
        offset_frame_length: how many frames should be marked as an offset during an offset.
        """
        if array_length is None:
            array_length = round(midi.get_end_time() / frame_length_seconds) + 1
        frames = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        onsets = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        offsets = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        velocities = np.zeros((array_length, config.midi_num_pitches), dtype=np.float32)
        pedals = np.zeros(array_length, dtype=np.float32)
        if config.extend_sustain_pedal:
            midi = MidiEncoding.extend_with_sustain(midi)

        pedal_events = MidiEncoding.get_sustain_events(midi)
        for event in pedal_events:
            start = round(event["start"] / frame_length_seconds)
            end = round(event["end"] / frame_length_seconds)
            pedals[start: end] = 1

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
            frame_length_seconds=frame_length_seconds,
            pedals=pedals
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
                    lambda x: x["start"] < note.end < x["end"],
                    sustain_events)
                )
                if len(sustain_end) > 0:
                    note.end = sustain_end[0]["end"]

                # check a new note starts before the end
                if i + 1 < len(note_bin) and note.end >= note_bin[i + 1].start:
                    note.end = note_bin[i + 1].start
        midi_copy.instruments[0].notes = list(sorted(
            itertools.chain.from_iterable(note_bins),
            key=lambda x: x.start))

        midi_copy.instruments[0].control_changes = []

        return midi_copy

    def decode(self, thresholds: Thresholds = None, vmin=0, vmax=127) -> Song:
        if thresholds is None:
            thresholds = Thresholds()

        pedal_events = []
        # Handle pedal events
        curr_pedal = False
        for i, pedal_p in enumerate(self.pedals):
            time = i
            if pedal_p > thresholds.pedal_on_threshold and not curr_pedal:
                curr_pedal = True
                pedal_events.append(PedalEvent(time=time, on=True))
            elif pedal_p < thresholds.pedal_off_threshold and curr_pedal:
                pedal_events.append(PedalEvent(time=time, on=False))
                curr_pedal = False

        # Handle note events
        onsets = self.onsets > thresholds.onset_threshold
        offsets = self.offsets > thresholds.offset_threshold
        frames = self.frames > thresholds.frame_threshold

        notes = []
        for pitch_id in range(config.midi_num_pitches):
            pitch_midi = pitch_id + config.midi_pitch_min
            onsets_for_pitch = onsets[:, pitch_id]
            onsets_for_pitch = keep_first_true(onsets_for_pitch)
            onset_frame_nums, = np.nonzero(onsets_for_pitch)
            curr_frame = 0
            for onset_frame in onset_frame_nums:
                if thresholds.min_note_length_frames >= 1 and not frames[curr_frame, pitch_id]:
                    # skip empty note
                    continue
                if curr_frame > onset_frame:
                    continue
                curr_frame = onset_frame

                # the note ends when {offsets} is true, or when {frames} is not true
                while curr_frame < offsets.shape[0]:
                    is_onset = onsets_for_pitch[curr_frame]
                    is_offset = offsets[curr_frame, pitch_id]
                    is_frame = frames[curr_frame, pitch_id]

                    if not is_onset and (is_offset or not is_frame):
                        # the note has ended
                        break
                    curr_frame += 1
                offset_frame = curr_frame
                if offset_frame - onset_frame < thresholds.min_note_length_frames:
                    # note too short, don't include it
                    continue
                onset_time = onset_frame
                offset_time = offset_frame
                velocity = int(self.velocities[onset_frame, pitch_id] * (vmax - vmin) + vmin)
                note = Note(start=onset_time, length=offset_time - onset_time, pitch=pitch_midi, velocity=velocity)
                notes.append(note)
        notes.sort(key=lambda note: note.start)
        return Song(
            notes=notes,
            pedal_events=pedal_events,
            frame_length_seconds=self.frame_length_seconds
        )

    def length_frames(self):
        return self.frames.shape[0]

    def cop_to_length(self, length):
        if length < self.length_frames():
            self.onsets = self.onsets[:length]
            self.offsets = self.offsets[:length]
            self.frames = self.frames[:length]
            self.velocities = self.velocities[:length]

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
            pedals=np.asarray(dict["pedals"]),
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
            "pedals": self.pedals
        }

    def plot_on_spectrogram(self, spectrogram):
        if spectrogram is not None:
            spectrogram = np.asarray(spectrogram)
            spectrogram = spectrogram.T  # take the transpose of the spectrogram, so it's plotted correctly
            librosa.display.specshow(spectrogram, sr=config.sample_rate, hop_length=config.hop_length, x_axis='time',
                                     y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
        frame_points = note_data_to_points(self.frames)
        plt.scatter(frame_points[0, :], frame_points[1, :], marker=",", s=3, alpha=0.5, color="blue")
        onset_points = note_data_to_points(self.velocities)
        plt.scatter(onset_points[0], onset_points[1], marker=",", s=3, alpha=0.5, color="green")
        offset_points = note_data_to_points(self.offsets)
        plt.scatter(offset_points[0, :], offset_points[1, :], marker=",", s=3, alpha=0.5, color="red")

        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(self.velocities.T, cmap='hot')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.title("Pedals")
        plt.plot(self.pedals)
        plt.plot()

    def plot(self, length=30):
        arr = np.array([self.offsets, self.onsets, self.frames])
        arr = np.swapaxes(arr, 0, 2)
        plt.figure(figsize=(length, 15))
        plt.imshow(arr, origin="lower")
        plt.show()

        plt.figure(figsize=(length, 6))
        plt.title("Pedals")
        plt.plot(self.pedals)
        plt.plot()
