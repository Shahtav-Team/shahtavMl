import madmom
import numpy as np
class BeatDownbeatDetector:
  def __init__(self):
    self.rnn_processor = madmom.features.RNNDownBeatProcessor()
  def find_beats(self, filename, allowed_beats_per_bar):
    # todo make this work with an already loaded file
    # should normalize audio to 16 bit so it works with library.
    dbn_processor = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar = allowed_beats_per_bar, fps = 100)
    rnn_result = self.rnn_processor(filename)
    beats_estimate = dbn_processor(rnn_result)
    is_downbeat = beats_estimate[:, 1] == 1
    downbeats = beats_estimate[is_downbeat][:, 0]
    beats = beats_estimate[:, 0]
    beats_per_bar = np.max(beats_estimate[:, 1])
    return beats, is_downbeat, beats_per_bar

beat_downbeat_detector = BeatDownbeatDetector()