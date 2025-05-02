# Customer Sentiment and Churn Analysis

This repository contains code and resources for analyzing customer sentiment and predicting churn using machine learning techniques. The project focuses on two datasets: Amazon Reviews and Telco Customer Churn.
The goal is to preprocess the data, train models, evaluate their performance, and explain the predictions using SHAP (SHapley Additive exPlanations).

## Datasets

1. **Amazon Reviews**: A dataset containing customer reviews from Amazon, which will be used for sentiment analysis. You can find the dataset [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
   - The dataset includes reviews, ratings, and other metadata about the products.
   - The sentiment analysis will classify reviews as positive or negative based on the rating.
   - The dataset is preprocessed to remove unnecessary columns and handle missing values.
   - The reviews are tokenized and padded to prepare them for input into the LSTM model.
   - The LSTM model is trained on the preprocessed reviews to predict sentiment.
   - The model is saved as `saved_lstm.h5` for future use.
2. **Telco Customer Churn**: A dataset containing customer information from a telecommunications company, which will be used for predicting customer churn. You can find the dataset [here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
   - The dataset includes customer demographics, account information, and whether the customer has churned or not.
   - The churn prediction model is trained using a Support Vector Machine (SVM) algorithm.
   - The model is saved as `saved_svm.pkl` for future use.
   - The dataset is preprocessed to handle categorical variables and missing values.

## Usage

> Make sure you have Python 3.9-3.11 installed. You can create a virtual environment to manage dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/shahriarpshuvo/customer-churn-prediction
   cd customer-sentiment-churn-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the datasets in the `data/` folder.

   ```sh
   project/
   ├── data/
   │   ├── amazon_reviews.csv
   │   └── telco_churn.csv
   ├── src/
   │   └── notebook.ipynb
   ├── models/
   ├── requirements.txt
   └── README.md
   ```

4. Run the notebook:

   ```bash
   jupyter notebook notebook.ipynb
   ```

## Results

The sentiment analysis model achieves an accuracy of 85% on the test set.
The churn prediction model achieves an accuracy of 78% on the test set.
The SHAP summary plot provides insights into the impact of different features on churn predictions.
