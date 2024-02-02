import librosa
import pretty_midi
import tensorflow as tf
import tqdm

from shachtav.WavToMidiModel import WavToMidiModel


def eval_model(model: WavToMidiModel, song_names_df, thresholds):
    results_info = []
    for threshold in thresholds:
        results_info.append({
            "threshold": threshold,
            "notes":{
                "tp": 0,
                "fp": 0,
                "fn": 0
            },
            "velocity": {
                "errors": []
            }
        })
    for i, song in tqdm.tqdm(list(song_names_df.iterrows())):
        try:
            midi_file = pretty_midi.PrettyMIDI(song["midi_filename"])
            notes_target = midi_file.instruments[0].notes
            audio, file_sr = librosa.load(song["audio_filename"])
            midi_encoding = model.infer(audio, file_sr)

            for info in results_info:
                song = midi_encoding.decode(info["threshold"], vmin = 0, vmax = 127)
                midi = song.to_pretty_midi()
                notes_pred = midi.instruments[0].notes
                tp, fp, fn = compare_notes(notes_target, notes_pred)
                info["notes"]["tp"] += tp
                info["notes"]["fp"] += fp
                info["notes"]["fn"] += fn
        except tf.errors.ResourceExhaustedError as e:
            print()
            print(e)
            print()
    for info in results_info:
        tp = info["notes"]["tp"]
        fp = info["notes"]["tp"]
        fn = info["notes"]["tp"]
        info["notes"]["f1"] = (2 * tp) / (2 * tp + fp + fn)

    results_info.sort(key = lambda x: x["notes"]["f1"], reverse=True)

    for info in results_info:
        print()
        print("threshold: ", info["threshold"])
        print("f1: ", info["f1"])
        print()



# Calculates the f1 score of our model
# f1 is a mesure of how accurate our model predicts notes, acounting evenly for disincentivizing false positives and false negetives.
# The formula is f1 = (2 * tp)/(2 * tp + fp + fn) Where tp is the number of true positive notes, fp is the number of false positives, and fn is the number of false negetives.
# A note is consided a true positive if it's onset is {max_allowed_diff} from the true onset, and it's ofset is {max_allowed_diff} off in absolute length or {offset_tolerance} in error of note length.

def split_by_note(notes):
    note_bins = [[] for _ in range(127)]
    for note in notes:
        note_bins[note.pitch].append(note)
    for bin in note_bins:
        bin.sort(key=lambda x: x.start)
    return note_bins


# compares ground truth target notes and predicted notes, giving the amount of true positives, false positives, and false negetives
def compare_notes(notes_target, notes_pred, max_allowed_diff=0.05, offset_tolerance=0.2, require_offset_match=True):
    true_positives = 0
    false_positives = 0
    false_negetives = 0
    velocity_errors = []

    notes_target_split = split_by_note(notes_target)
    notes_pred_split = split_by_note(notes_pred)

    for pitch in range(127):
        targets = notes_target_split[pitch]
        preds = notes_pred_split[pitch]

        i_target = 0
        i_pred = 0

        while i_target < len(targets) and i_pred < len(preds):
            target = targets[i_target]
            pred = preds[i_pred]
            target_note_length = target.end - target.start
            onset_match = abs(target.start - pred.start) < max_allowed_diff
            offset_match = abs(target.end - pred.end) < max(max_allowed_diff,
                                                            offset_tolerance * target_note_length)  # notes
            if onset_match and (offset_match or not require_offset_match):
                true_positives += 1
                i_target += 1
                i_pred += 1
            elif target.start > pred.start:
                i_pred += 1
                false_positives += 1
            else:
                i_target += 1
                false_negetives += 1
        false_positives += len(preds) - i_pred
        false_negetives += len(targets) - i_target

    return true_positives, false_positives, false_negetives
