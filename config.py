sample_rate = 16000
hop_length = 512
n_fft = 2048
spectrogram_n_bins = 229
spectrogram_f_min = 30
spectrogram_f_max = 8000
# Number of seconds in each chunk entered into the neural network
chunk_length_seconds = 12

midi_pitch_min = 21
midi_pitch_max = 108
midi_num_pitches = 88

encoding_onset_length_frames = 2
encoding_offset_length_frames = 2

extend_sustain_pedal = False

frame_length_seconds = hop_length / sample_rate
chunk_length_frames = chunk_length_seconds / frame_length_seconds
# assert that chunk_length_frames is a whole number, meaning the chunk length perfectly divides the frame length.
assert chunk_length_frames == int(chunk_length_frames)
chunk_length_frames = int(chunk_length_frames)
raw_audio_chunk_length = chunk_length_seconds * sample_rate