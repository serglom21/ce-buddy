# sentry_suggestion/train.py
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import pickle
import os

def load_and_preprocess_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    code = [item['code'] for item in data]
    description = [item['description'] for item in data]
    sentry_code = [item['sentry_code'] for item in data]

    # Add start and end tokens
    sentry_code_with_tokens = ["<s> " + code + " </s>" for code in sentry_code]

    input_texts = [c + " " + d for c, d in zip(code, description)]

    def tokenize(texts, num_words=None):
        tokenizer = Tokenizer(num_words=num_words, filters='')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        return tokenizer, sequences

    input_tokenizer, input_sequences = tokenize(input_texts)
    output_tokenizer, output_sequences = tokenize(sentry_code_with_tokens) # changed to use sentry_code_with_tokens

    max_input_len = max(len(s) for s in input_sequences)
    max_output_len = max(len(s) for s in output_sequences)

    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
    padded_output_sequences = pad_sequences(output_sequences, maxlen=max_output_len, padding='post')

    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1

    X_train, X_test, y_train, y_test = train_test_split(padded_input_sequences, padded_output_sequences, test_size=0.2, random_state=42)

    return X_train, y_train, input_tokenizer, output_tokenizer, max_input_len, max_output_len, input_vocab_size, output_vocab_size

def build_and_train_model(X_train, y_train, input_vocab_size, output_vocab_size, max_input_len, max_output_len, epochs=50):
    embedding_dim = 256
    latent_dim = 512

    encoder_inputs = Input(shape=(max_input_len,))
    encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_output_len,))
    decoder_embedding = Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    decoder_input_data = np.zeros_like(y_train)
    decoder_input_data[:, 1:] = y_train[:, :-1]
    decoder_input_data[:, 0] = output_tokenizer.word_index['<s>'] #start token
    decoder_target_data = np.expand_dims(y_train, -1)

    model.fit([X_train, decoder_input_data], decoder_target_data, batch_size=64, epochs=epochs, validation_split=0.2)

    return model

# Load and preprocess
X_train, y_train, input_tokenizer, output_tokenizer, max_input_len, max_output_len, input_vocab_size, output_vocab_size = load_and_preprocess_data('data/data.json')

# Train the model
model = build_and_train_model(X_train, y_train, input_vocab_size, output_vocab_size, max_input_len, max_output_len)

# Save the model and tokenizers
os.makedirs('models', exist_ok=True)
model.save('models/sentry_suggestion_model.h5')
os.makedirs('tokenizers', exist_ok=True)
with open('tokenizers/input_tokenizer.pkl', 'wb') as f:
    pickle.dump(input_tokenizer, f)
with open('tokenizers/output_tokenizer.pkl', 'wb') as f:
    pickle.dump(output_tokenizer, f)
with open('tokenizers/model_metadata.pkl', 'wb') as f:
    pickle.dump({'max_input_len': max_input_len, 'max_output_len': max_output_len}, f)

print("Model training complete. Model and tokenizers saved.")