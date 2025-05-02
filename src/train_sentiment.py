"""
train_sentiment.py
------------------
Train sentiment classification models: SVM and LSTM
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

# ========== Load Data ==========
train_df = pd.read_csv("../data/sentiment_train.csv")
val_df = pd.read_csv("../data/sentiment_val.csv")
test_df = pd.read_csv("../data/sentiment_test.csv")

# ========== SVM Pipeline ==========
def train_svm(train_df, test_df):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df["clean_text"])
    X_test = tfidf.transform(test_df["clean_text"])

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    svm = LinearSVC()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(svm, "../models/saved_svm.pkl")
    joblib.dump(tfidf, "../models/saved_tfidf.pkl")

# ========== LSTM Pipeline ==========
def train_lstm(train_df, val_df, test_df):
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = tokenizer.texts_to_sequences(train_df["clean_text"])
    X_val = tokenizer.texts_to_sequences(val_df["clean_text"])
    X_test = tokenizer.texts_to_sequences(test_df["clean_text"])

    max_len = 100
    X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
    X_val = pad_sequences(X_val, maxlen=max_len, padding="post")
    X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

    le = LabelEncoder()
    y_train = to_categorical(le.fit_transform(train_df["label"]))
    y_val = to_categorical(le.transform(val_df["label"]))
    y_test = to_categorical(le.transform(test_df["label"]))

    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=64, input_length=max_len))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

    model.evaluate(X_test, y_test)
    model.save("../models/saved_lstm.h5")
    joblib.dump(tokenizer, "../models/saved_tokenizer.pkl")

if __name__ == "__main__":
    train_svm(train_df, test_df)
    train_lstm(train_df, val_df, test_df)
    print("✅ Sentiment models trained and saved.")
