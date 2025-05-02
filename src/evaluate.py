"""
evaluate.py
-----------
Evaluate sentiment and churn prediction models.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load test data
sentiment_test = pd.read_csv("../data/sentiment_test.csv")
churn_test = pd.read_csv("../data/churn_test.csv")

# ========== Sentiment Model Evaluation ==========

def evaluate_svm():
    tfidf = joblib.load("../models/saved_tfidf.pkl")
    svm = joblib.load("../models/saved_svm.pkl")

    X_test = tfidf.transform(sentiment_test["clean_text"])
    le = LabelEncoder()
    y_true = le.fit_transform(sentiment_test["label"])
    y_pred = svm.predict(X_test)

    print("SVM Sentiment Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

def evaluate_lstm():
    tokenizer = joblib.load("../models/saved_tokenizer.pkl")
    model = load_model("../models/saved_lstm.h5")

    sequences = tokenizer.texts_to_sequences(sentiment_test["clean_text"])
    X_test = pad_sequences(sequences, maxlen=100)
    le = LabelEncoder()
    y_true = le.fit_transform(sentiment_test["label"])
    y_true_cat = to_categorical(y_true)

    loss, acc = model.evaluate(X_test, y_true_cat, verbose=0)
    print(f"LSTM Sentiment Accuracy: {acc:.4f}")

# ========== Churn Model Evaluation ==========

def evaluate_churn(model_path, name="Model"):
    model = joblib.load(model_path)
    scaler = joblib.load("../models/saved_scaler.pkl")
    X_test = churn_test.drop(columns=["Churn"])
    y_true = churn_test["Churn"]
    X_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print(f"{name} Churn Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")

if __name__ == "__main__":
    evaluate_svm()
    evaluate_lstm()
    evaluate_churn("../models/saved_churn_lr.pkl", "Logistic Regression")
    evaluate_churn("../models/saved_churn_rf.pkl", "Random Forest")
    print("✅ Evaluation complete.")
