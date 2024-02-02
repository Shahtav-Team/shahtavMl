from dataclasses import dataclass
from typing import List

import music21
import numpy as np
import pretty_midi

from shachtav.BeatDownbeatDetector import BeatInfo
from shachtav.VoiceSplitting import NotesSplit


@dataclass
class NoteGroup:
    start_time: int
    notes: list[pretty_midi.Note]

def downbeat_times(downbeats):
    downbeat_indexes, = np.nonzero(downbeats)
    return downbeat_indexes * 4

def find_next_downbeat(db_times, time):
    next_db_index = np.searchsorted(db_times, time, side = "right")
    if next_db_index < len(db_times):
        return db_times[next_db_index]
    else:
        # todo: figure out a proper number to put here based on the time signature.
        return db_times[-1] + 16

def notes_to_music21(notes: List[pretty_midi.Note],
                     stream: music21.stream.Stream,
                     beat_info: BeatInfo,
                     beat_phase):
    notes = sorted(notes, key=lambda note: note.start)
    # group notes by the same start time
    i = 0
    notes_groups = [NoteGroup(start_time=0, notes=[])]
    for note in notes:
        if note.start == notes_groups[-1].start_time:
            notes_groups[-1].notes.append(note)
        else:
            notes_groups.append(NoteGroup(start_time=note.start, notes=[note]))

    time_signature = music21.meter.base.TimeSignature()
    time_signature.numerator = int(beat_info.beats_per_bar)
    time_signature.denominator = 4
    stream.append(time_signature)
    # add all the notes to the stream
    for i, group in enumerate(notes_groups):
        if len(group.notes) == 0:
            continue
        start_time = group.start_time

        longest_duration = max((note.duration) for note in group.notes)
        # remove notes significantly longer than the max duration, because they are likely noise or irrelevant.
        if longest_duration >= 2:
            valid_notes = filter(lambda note: note.duration * 2 >= longest_duration, group.notes)
        else:
            valid_notes = group.notes

        if i + 1 < len(notes_groups):
            next_note_start = notes_groups[i + 1].start_time
        else:
            # todo: figure out exactly what we're doing with the last note, since we can't use the next note for refrence of length.
            next_note_start = start_time + 4

        note_end = next_note_start

        pitches = [note.pitch for note in valid_notes]
        length_quarters = (note_end - start_time) / 4  # divide by 4, since we are measuring in 16ths and need quarters.
        if len(pitches) == 0:
            to_add = music21.note.Rest(quarterLength=length_quarters)
            stream.insert(start_time / 4 + beat_phase, to_add)
        else:
            to_add = music21.chord.Chord([music21.note.Note(pitch=pitch, quarterLength=length_quarters) for pitch in pitches], quarterLength=length_quarters)
            stream.insert(start_time / 4 + beat_phase, to_add)

    # For some reason, natural notes are given accidentals even though they shouldn't be given.
    # Remove those accidentals.
    for note in stream.notes:
        for pitch in note.pitches:
            if pitch.accidental is not None and pitch.accidental.alter == 0:
                pitch.accidental = None
    # add measures based on the given beat.
    stream.makeMeasures(inPlace=True)
    # Realign to the beat
    mes1 = stream.measure(1)
    mes1.shiftElements(-beat_phase, startOffset=1/8)
    mes1.paddingLeft = beat_phase
    stream.shiftElements(-beat_phase, startOffset=1/8)

    # For notes that span between multiple measures, truncate them at the end of their first measure
    for measure in stream.getElementsByClass(music21.stream.Measure):
        measure_length = measure.barDuration.quarterLength - measure.paddingLeft
        for note in measure.notes:
            if note.offset + note.quarterLength > measure_length:
                note.quarterLength = measure_length - note.offset
                measure.clearCache()

    stream.makeRests(inPlace=True, timeRangeFromBarDuration=True)

def score_from_notes(notes_split: NotesSplit, beat_info: BeatInfo,
                     score_title="Title",
                     score_composer="Composer"):
    """
    Produces a music21 score object from info from other models.
    Args:
        notes_split: from ClefSplitModel
        beat_info: from BeatDownbeatDetector
    Returns:

    Raises:
        Exception if something goes wrong. Should wrap this with try and catch.
    """
    if notes_split is None:
        raise ValueError("notes_split must not be None")

    # the start of the piece is not always a downbeat.
    # Shift the notes so they align properly with the downbeats.
    beat_phase = np.argmax(beat_info.is_downbeat)

    score = music21.stream.Score()
    treble_part = music21.stream.Part(id="treble")
    treble_part.append(music21.clef.TrebleClef())
    notes_to_music21(notes_split.treble_notes, treble_part, beat_info, beat_phase)

    bass_part = music21.stream.Part(id="bass")
    bass_part.append(music21.clef.BassClef())
    notes_to_music21(notes_split.bass_notes, bass_part, beat_info, beat_phase)

    score.insert(0, treble_part)
    score.insert(0, bass_part)

    key = score.analyze("key")
    score.parts[0].measure(1).insert(0, key)
    score.parts[1].measure(1).insert(0, key)
    music21.stream.makeNotation.makeAccidentalsInMeasureStream(score.parts[0], useKeySignature=key)
    music21.stream.makeNotation.makeAccidentalsInMeasureStream(score.parts[1], useKeySignature=key)

    # score.insert(0, music21.metadata.Metadata())
    # score.metadata.title = score_title
    # score.metadata.composer = score_composer

    return score