import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

def main():
    model_path = 'text_model.h5'
    tokenizer_path = 'tokenizer.pickle'
    max_sequence_len = 20  # This should match the max_sequence_len used during training

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    print("Model and tokenizer loaded. Enter a seed text to generate the next word.")

    while True:
        seed_text = input("Enter seed text: ")
        if seed_text.lower() == 'exit':
            print("Exiting...")
            break
        next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
        if next_word:
            print(f"Next word prediction: {next_word}")
        else:
            print("Could not predict the next word. Please try again with a different seed text.")

if __name__ == "__main__":
    main()
