import pretty_midi
import tensorflow as tf

MAX_TIME_VALUE = 500
SUSTAIN_NO = 64


def event_list_to_tokens(events):
    events.sort(key=lambda e: (e["time"], e["type"]))

    # the current bin used for an event
    current_bin = 0

    # the last event type
    current_event_type = None

    for event in events:
        event_bin = int(event["time"] * 100)

        if event_bin != current_bin:
            time_difference = (event_bin - current_bin)

            while time_difference > 0:
                yield f"<time {min(time_difference, MAX_TIME_VALUE) / 100.0}>"
                time_difference -= MAX_TIME_VALUE

            current_bin = event_bin
            current_event_type = None

        if current_event_type != event["type"]:
            current_event_type = event["type"]
            yield f"<{current_event_type}>"

        if "velocity" in event:
            yield f"<velocity {event['velocity']}>"
        if "note" in event:
            yield f"<note {event['note']}>"


def convert_to_event_list(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)

    events = []

    for note in midi.instruments[0].notes:
        events += [
            {
                "note": note.pitch,
                "type": "on",
                "time": note.start,
                "velocity": note.velocity
            },
            {
                "note": note.pitch,
                "type": "off",
                "time": note.end,
                "velocity": 0
            }
            ]
    for control_change in midi.instruments[0].control_changes:
        if control_change.number == SUSTAIN_NO:
            events.append({
                "type": "sustain" + ("_on" if control_change.value > 64 else "_off"),
                "time": control_change.time
            })

    return events


def main():
    events = convert_to_event_list("../samples/test.mid")

    for token in event_list_to_tokens(events):
        print(token)


if __name__ == "__main__":
    print(tf.version.VERSION)
