# 💰 Loan Approval Prediction App

This is a real-world AI-powered application that predicts whether a loan applicant will be approved, partially approved, or denied based on financial data. It uses an explainable machine learning model (XGBoost + SHAP) and is deployed using Streamlit.

## Features

- Predict loan approval status (100%, 75%, 50%, or denied)
- Dynamic SHAP-based explainability for full transparency
- Engineered smart financial metrics like `EMI`, `Total_Income`, `Income_To_Loan`
- Optimized with `RandomizedSearchCV` for high accuracy
- Built-in Streamlit interface for real-time user interaction

## Model & Tech Stack

- **Algorithm**: XGBoostClassifier (with hyperparameter tuning)
- **Feature Engineering**: EMI, Income Ratios, Total Income
- **Explainability**: SHAP values (summary)
- **Web App**: Streamlit (Multi-page UI)
- **Other Tools**: Scikit-learn, Pandas, Matplotlib, Seaborn

## Dataset

Kaggle Dataset: [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

##  Live Demo

🔗 [Streamlit App Link](https://loan-approval-app-6tzfz8hvvqdmc5czo9l7vc.streamlit.app/)

## 🛠 How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/your-username/loan-approval-app.git

# 2. Navigate into the project
cd loan-approval-app

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

