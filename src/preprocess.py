"""
preprocess.py
-------------
Unified preprocessing for:
1. Amazon Fine Food Reviews (sentiment analysis)
2. Telco Customer Churn (churn prediction)
"""

import pandas as pd
import numpy as np
import re
import string
import os
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ========== SENTIMENT ANALYSIS PREPROCESSING ==========

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text, lemmatizer, stop_words):
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(lemmatized)

def preprocess_amazon_reviews(input_path, output_dir):
    df = pd.read_csv(input_path)
    df = df[["Text", "Score"]].dropna()
    df = df[df["Score"].isin([1, 2, 3, 4, 5])]
    df["label"] = df["Score"].apply(lambda x: "negative" if x <= 2 else "neutral" if x == 3 else "positive")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df["clean_text"] = df["Text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].apply(lambda x: lemmatize_text(x, lemmatizer, stop_words))

    df_final = df[["clean_text", "label"]]

    train, val_test = train_test_split(df_final, test_size=0.3, stratify=df_final["label"], random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, stratify=val_test["label"], random_state=42)

    train.to_csv(os.path.join(output_dir, "sentiment_train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "sentiment_val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "sentiment_test.csv"), index=False)

# ========== CHURN PREDICTION PREPROCESSING ==========

def preprocess_telco_churn(input_path, output_dir):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col != "customerID"]
    df = df.drop(columns=["customerID"])

    df = pd.get_dummies(df, columns=cat_cols)

    train, val_test = train_test_split(df, test_size=0.3, stratify=df["Churn"], random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, stratify=val_test["Churn"], random_state=42)

    train.to_csv(os.path.join(output_dir, "churn_train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "churn_val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "churn_test.csv"), index=False)

if __name__ == "__main__":
    preprocess_amazon_reviews("../data/amazon_reviews.csv", "../data")
    preprocess_telco_churn("../data/telco_churn.csv", "../data")
    print("✅ Preprocessing complete for both datasets.")
