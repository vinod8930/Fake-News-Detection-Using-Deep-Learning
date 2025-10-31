import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import zipfile



# File paths (update to local paths)

train_zip_path = "C:/Users/vinod/Desktop/DL PROJECT/train.csv"

test_zip_path = "C:/Users/vinod/Desktop/DL PROJECT/test.csv"

submit_csv_path = "submit.csv"



# Extract files

train_csv_path = "train.csv"

test_csv_path = "test.csv"



with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:

    zip_ref.extractall()

    

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:

    zip_ref.extractall()



# Load datasets

train_df = pd.read_csv(train_csv_path)

test_df = pd.read_csv(test_csv_path)



# Preprocessing

texts = train_df['text'].astype(str).values

labels = train_df['label'].values  # 1 for fake, 0 for real



# Tokenization and padding

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')



# Train-test split

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)



# Model building

model = Sequential([

    Embedding(10000, 128, input_length=300),

    Bidirectional(LSTM(64, return_sequences=True)),

    Dropout(0.3),

    Bidirectional(LSTM(64)),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')

])



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Train model

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))



# Evaluate model

y_pred = (model.predict(X_test) > 0.5).astype('int32')

print("Accuracy:", accuracy_score(y_test, y_pred))



# Predict on test dataset

test_texts = test_df['text'].astype(str).values

test_sequences = tokenizer.texts_to_sequences(test_texts)

test_padded = pad_sequences(test_sequences, maxlen=300, padding='post', truncating='post')

test_predictions = (model.predict(test_padded) > 0.5).astype('int32')



# Save predictions

submission = pd.DataFrame({

    'id': test_df['id'],

    'label': test_predictions.flatten()

})

submission.to_csv(submit_csv_path, index=False)

print("Predictions saved to submit.csv")