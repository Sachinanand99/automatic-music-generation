import os
import pickle
import numpy as np
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from mealpy.swarm_based import GWO
from music21 import stream, note, chord, midi, instrument
from sklearn.model_selection import train_test_split

# --------------------- Load Notes ---------------------
def load_notes(file_path="notes.pkl"):
    with open(file_path, "rb") as f:
        notes = pickle.load(f)
    return notes

# --------------------- Prepare Sequences ---------------------
def prepare_sequences(notes, sequence_length=100):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = to_categorical(network_output)

    return network_input, network_output, pitchnames

# --------------------- Build Model ---------------------
def build_model(input_shape, output_size, lstm_units=512, dropout_rate=0.3):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dense(256),
        Dropout(dropout_rate),
        Dense(output_size),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# --------------------- GWO Optimization ---------------------
def optimize_with_gwo(network_input, network_output):
    def fitness_func(solution):
        lstm_units = int(solution[0])
        dropout = float(solution[1])
        model = build_model(network_input.shape[1:], network_output.shape[1], lstm_units, dropout)
        history = model.fit(network_input, network_output, epochs=2, batch_size=64, verbose=0)
        return history.history['loss'][-1]  # Minimize loss

    problem = {
        "fit_func": fitness_func,
        "lb": [128, 0.1],
        "ub": [1024, 0.5],
        "minmax": "min",
        "log_to_file": False,
        "verbose": False
    }
    model_gwo = GWO.OriginalGWO(epoch=5, pop_size=5)
    best = model_gwo.solve(problem)
    best_lstm, best_dropout = int(best.best_position[0]), float(best.best_position[1])
    return best_lstm, best_dropout

# --------------------- Train and Time ---------------------
def train_model(model, X, y, epochs=20, label="Model", save_path=None):
    print(f"\nTraining {label}...")
    start = time.time()
    model.fit(X, y, epochs=epochs, batch_size=64)
    end = time.time()
    print(f"{label} training completed in {(end - start):.2f} seconds.")
    if save_path:
        model.save(save_path)
        print(f"{label} saved to {save_path}")
    return model

# --------------------- Evaluate Model ---------------------
def evaluate_model(model, X_test, y_test, label="Model"):
    print(f"\nEvaluating {label}...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{label} Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")

# --------------------- Generate Music ---------------------
def generate_notes(model, network_input, pitchnames, note_count=500, output_txt="generated_notes.txt", output_midi="generated.mid"):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start_index = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_index]
    prediction_output = []

    for _ in range(note_count):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], [[index / float(len(pitchnames))]], axis=0)

    with open(output_txt, "w") as f:
        for note in prediction_output:
            f.write(f"{note}\n")
    print(f"\nGenerated notes saved to {output_txt}")

    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in chord_notes:
                n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(chord_notes)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_midi)
    print(f"Generated MIDI saved to {output_midi}")

# --------------------- Main ---------------------
def main():
    print("Loading notes...")
    notes = load_notes()
    print(f"Loaded {len(notes)} notes.")
    print("Preparing sequences...")
    X, y, pitchnames = prepare_sequences(notes)
    print(f"Prepared {len(X)} sequences.")
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Without GWO
    print("Building basic LSTM model...")
    model_basic = build_model(X.shape[1:], y.shape[1])
    train_model(model_basic, X_train, y_train, label="Basic LSTM", save_path="model_basic.h5")
    evaluate_model(model_basic, X_test, y_test, label="Basic LSTM")
    generate_notes(model_basic, X, pitchnames, output_txt="generated_basic.txt", output_midi="generated_basic.mid")

    # With GWO
    print("Building model with GWO...")
    lstm_units, dropout_rate = optimize_with_gwo(X_train, y_train)
    model_gwo = build_model(X.shape[1:], y.shape[1], lstm_units, dropout_rate)
    train_model(model_gwo, X_train, y_train, label=f"GWO LSTM ({lstm_units} units, {dropout_rate:.2f} dropout)", save_path="model_gwo.h5")
    evaluate_model(model_gwo, X_test, y_test, label="GWO LSTM")
    generate_notes(model_gwo, X, pitchnames, output_txt="generated_gwo.txt", output_midi="generated_gwo.mid")

if __name__ == '__main__':
    main()
