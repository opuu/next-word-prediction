import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import Flask, request, jsonify
import os
import pickle

def load_text_data(data_folder):
    texts = ""
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts += file.read().lower() + "\n"
    return texts

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Load the trained model
model = tf.keras.models.load_model('text_model.h5')

# Load the tokenizer
tokenizer = load_tokenizer('tokenizer.pickle')

# Load text data for determining max_sequence_len
text = load_text_data('data')
input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

# Flask web server
app = Flask(__name__)

def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)
        output_word = tokenizer.index_word.get(predicted_word_index[0], '')
        seed_text += " " + output_word
    return seed_text

@app.route('/', methods=['GET'])
def autocomplete():
    seed_text = request.args.get('sentence', default="", type=str)
    if not seed_text:
        return jsonify([])

    next_words = 3
    generated_text = generate_text(seed_text, next_words, max_sequence_len)
    response_words = generated_text.split()[-next_words:]
    return jsonify(response_words)

if __name__ == "__main__":
    app.run(debug=True)
