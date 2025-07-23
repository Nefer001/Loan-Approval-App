import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pickle

# Load + clean
df = pd.read_csv("Loan Prediction.csv")
df.fillna({
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
    'Credit_History': df['Credit_History'].median()
}, inplace=True)

for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = LabelEncoder().fit_transform(df[col])
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

# Engineering
df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_To_Loan'] = df['Total_Income'] / (df['LoanAmount'] + 1e-6)

# Train
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'Total_Income', 'EMI', 'Income_To_Loan']]
Y = df['Loan_Status']

X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', class_weight='balanced')
param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
}
search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5, scoring='accuracy')
search.fit(X_train_scaled, Y_train)

# Save
with open("loan_model.pkl", "wb") as f:
    pickle.dump((search.best_estimator_, scaler), f)
