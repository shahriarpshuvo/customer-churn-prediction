"""
train_churn.py
--------------
Train a churn prediction model using usage data and optional sentiment features.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    train = pd.read_csv("../data/churn_train.csv")
    val = pd.read_csv("../data/churn_val.csv")
    test = pd.read_csv("../data/churn_test.csv")

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_val = val.drop(columns=["Churn"])
    y_val = val["Churn"]

    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, "../models/saved_churn_rf.pkl")

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, "../models/saved_churn_lr.pkl")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_prepare_data()
    train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test)
    joblib.dump(scaler, "../models/saved_scaler.pkl")
    print("✅ Churn models trained and saved.")
