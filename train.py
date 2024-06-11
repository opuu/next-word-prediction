import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np
import os
import pickle

def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()

def generate_sequences(tokenizer, text, max_sequence_len):
    input_sequences = []
    for line in text.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def create_dataset(file_paths, tokenizer, max_sequence_len, buffer_size=10000, batch_size=32):
    def generator():
        for file_path in file_paths:
            text = load_text_data(file_path)
            sequences = generate_sequences(tokenizer, text, max_sequence_len)
            for sequence in sequences:
                padded_sequence = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')[0]
                predictors, label = padded_sequence[:-1], padded_sequence[-1]
                yield predictors, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(max_sequence_len-1,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    # Load and preprocess data
    data_folder = 'data'
    file_paths = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.endswith(".txt")]

    # Fit tokenizer on the entire dataset
    tokenizer = Tokenizer()
    all_text = ""
    for file_path in file_paths:
        all_text += load_text_data(file_path) + "\n"
    tokenizer.fit_on_texts([all_text])
    total_words = len(tokenizer.word_index) + 1

    # Prepare input sequences
    max_sequence_len = 20  # Maximum length of sequences (adjust as needed)
    dataset = create_dataset(file_paths, tokenizer, max_sequence_len)

    # Define the model
    def create_model():
        model = Sequential()
        model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(total_words, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Detect and use the best available hardware
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU")
    except ValueError:
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            print("Running on GPU")
        else:
            strategy = tf.distribute.get_strategy()
            print("Running on CPU")

    with strategy.scope():
        model = create_model()
        model.fit(dataset, epochs=20, verbose=1)

    # Save the model and tokenizer
    model.save('text_model.h5')

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
