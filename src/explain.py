"""
explain.py
----------
Explain churn predictions using SHAP.
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load test data and model
df = pd.read_csv("../data/churn_test.csv")
model = joblib.load("../models/saved_churn_rf.pkl")
scaler = joblib.load("../models/saved_scaler.pkl")

# Prepare data
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_scaled = scaler.transform(X)

# Use TreeExplainer for Random Forest
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Summary Plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("../models/shap_summary_plot.png")
print("✅ SHAP summary plot saved to models/shap_summary_plot.png")
