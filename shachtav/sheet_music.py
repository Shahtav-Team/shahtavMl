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

def notes_to_music21(notes: List[pretty_midi.Note], stream: music21.stream.Stream, beat_info: BeatInfo):
    notes = sorted(notes, key=lambda note: note.start)
    db_times = downbeat_times(beat_info.is_downbeat)
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
    # the start of the piece is not always a downbeat.
    # add rests at the start of the beat so that the start of each coresponds with a downbeat.
    beat_phase = np.argmax(beat_info.is_downbeat)
    if beat_phase != 0:
        stream.append(music21.note.Rest(beat_phase))
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

        next_downbeat = find_next_downbeat(db_times, start_time)

        note_end = min(next_note_start, next_downbeat) # notes can't last between multiple measures.
        note_end = max(note_end, next_note_start) # ensure notes don't have negative duration

        pitches = [note.pitch for note in valid_notes]
        length_quarters = (note_end - start_time) / 4  # divide by 4, since we are measuring in 16ths and need quarters.
        if len(pitches) == 0:
            to_add = music21.note.Rest(quarterLength=length_quarters)
            stream.insert(start_time / 4, to_add)
        else:
            to_add = music21.chord.Chord(pitches, quarterLength=length_quarters)
            stream.insert(start_time / 4, to_add)

    # add measures based on the given beat.
    stream.makeMeasures(inPlace=True)
    stream.makeRests(inPlace=True)


def score_from_notes(notes_split: NotesSplit, beat_info: BeatInfo):
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
    score = music21.stream.Score()
    treble_part = music21.stream.Part(id="treble")
    treble_part.append(music21.clef.TrebleClef())
    notes_to_music21(notes_split.treble_notes, treble_part, beat_info)

    bass_part = music21.stream.Part(id="bass")
    bass_part.append(music21.clef.BassClef())
    notes_to_music21(notes_split.bass_notes, bass_part, beat_info)

    score.insert(0, treble_part)
    score.insert(1, bass_part)

    key = score.analyze("key")
    score.parts[0].measure(1).insert(0, key)
    score.parts[1].measure(1).insert(0, key)

    return score
