import music21
from dataclasses import dataclass
import numpy as np
import pretty_midi
from shachtav.BeatDownbeatDetector import BeatInfo
from . import VoiceSplitting


@dataclass
class NoteGroup:
  start_time: int
  notes: list[pretty_midi.Note]


def notes_to_music21(notes, stream, beat_info: BeatInfo):
  notes = sorted(notes, key = lambda note: note.start)

  # group notes by the same start time
  i = 0
  notes_groups = [NoteGroup(start_time = 0, notes=[])]
  for note in notes:
    if note.start == notes_groups[-1].start_time:
      notes_groups[-1].notes.append(note)
    else:
      notes_groups.append(NoteGroup(start_time = note.start, notes=[note]))

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
    start_time = group.start_time
    pitches = [note.pitch for note in group.notes]
    if i + 1 < len(notes_groups):
      end_time = notes_groups[i + 1].start_time
    else:
      end_time = start_time + 4 # todo: figure out exactly what we're doing with the last note, since we can't use the next note for refrence of length.

    length_quarters = (end_time - start_time) / 4 # divide by 4, since we are mesuring in 16ths and need quarters.
    if len(pitches) == 0:
      to_add = music21.note.Rest(quarterLength = length_quarters)
      stream.append(to_add)
    else:
      to_add = music21.chord.Chord(pitches, quarterLength = length_quarters)
      stream.append(to_add)

  key = stream.analyze("key")
  stream.insert(0, key)


  # add measures based on the given beat.
  stream.makeMeasures(inPlace = True)