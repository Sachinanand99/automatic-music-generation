from music21 import converter, instrument, note, chord
import os
import pickle
import glob

def extract_notes(data_dir, save_every=100):
    notes = []
    midi_files = glob.glob(os.path.join(data_dir, '**/*.mid*'), recursive=True)

    print(f"Found {len(midi_files)} MIDI files.\n")

    for i, file in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}] Processing: {os.path.basename(file)}")
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                elements = parts.parts[0].recurse()
            else:
                elements = midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        except Exception as e:
            print(f"Error parsing {file}: {e}")

        # Save checkpoint every N files
        if i % save_every == 0 or i == len(midi_files):
            with open("notes_checkpoint.pkl", "wb") as f:
                pickle.dump(notes, f)
            print(f"Saved progress at {i} files with {len(notes)} notes.")

    print(f"\nTotal notes extracted: {len(notes)}")

    # Delete checkpoint file after operation
    if os.path.exists("notes_checkpoint.pkl"):
        os.remove("notes_checkpoint.pkl")
        print("Deleted temporary checkpoint file.")

    return notes

notes = extract_notes("maestro-v3.0.0")

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)
