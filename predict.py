# sentry_suggestion/predict.py
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys

def load_model_and_tokenizers():
    model = tf.keras.models.load_model('models/sentry_suggestion_model.h5')
    with open('tokenizers/input_tokenizer.pkl', 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open('tokenizers/output_tokenizer.pkl', 'rb') as f:
        output_tokenizer = pickle.load(f)
    with open('tokenizers/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    max_input_len = metadata['max_input_len']
    max_output_len = metadata['max_output_len']
    return model, input_tokenizer, output_tokenizer, max_input_len, max_output_len

def generate_sentry_suggestion(input_text, model, input_tokenizer, output_tokenizer, max_input_len, max_output_len):
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    padded_input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    encoder_inputs = model.input[0]
    encoder_outputs, state_h, state_c = model.layers[4].output
    encoder_model = tf.keras.Model(encoder_inputs, [state_h, state_c])
    states_value = encoder_model.predict(padded_input_seq)

    decoder_inputs = tf.keras.Input(shape=(1,)) #create new input layer
    decoder_state_input_h = tf.keras.Input(shape=(512,)) #create new input layer
    decoder_state_input_c = tf.keras.Input(shape=(512,)) #create new input layer
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding = model.layers[3]
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index['<s>']

    decoded_sentence = []
    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == '</s>':
            break
        decoded_sentence.append(sampled_word)
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return ' '.join(decoded_sentence)

def is_code_snippet(text):
    code_keywords = ["function", "def", "class", "import", "from", "async", "await", "=>"]
    for keyword in code_keywords:
        if keyword in text.lower():
            return True
    return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"<input text or code snippet>\"")
        sys.exit(1)

    input_text = sys.argv[1]

    if is_code_snippet(input_text):
        processed_input = input_text
    else:
        processed_input = f"// {input_text}. Description: {input_text}."

    model, input_tokenizer, output_tokenizer, max_input_len, max_output_len = load_model_and_tokenizers()
    suggestion = generate_sentry_suggestion(processed_input, model, input_tokenizer, output_tokenizer, max_input_len, max_output_len)

    print("Sentry Suggestion:", suggestion)