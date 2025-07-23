import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import streamlit as st
import seaborn as sns
import shap
import pickle
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Loan Approval Predictor', layout='wide')

X_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area',
            'Total_Income', 'EMI', 'Income_To_Loan']
# Load pre-trained model + scaler
@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Loan Prediction.csv')

    # Fill Nulls
    df.fillna({
        'LoanAmount': df['LoanAmount'].median(),
        'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
        'Credit_History': df['Credit_History'].median()
    }, inplace=True)

    # Encode Categorical
    encoder = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[col] = encoder.fit_transform(df[col])

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

    # Feature Engineering
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_To_Loan'] = df['Total_Income'] / (df['LoanAmount'] + 1e-6)

    return df

df = load_data()

# ---------------- UI ---------------- #

st.sidebar.header('🧭 Navigation Pages')
pages = st.sidebar.radio('Select Page', ['Home', 'Data Info', 'Prediction Approval'])

if pages == 'Home':
    st.title('Loan Approval Predictor 💰')
    st.markdown("""
    Welcome to The Loan Approval Prediction App!  
    This tool uses a powerful AI model (XGBoost) to predict whether a loan application is likely to be approved or not.  
    Based on a [real-world financial dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

    **What You Can Do:**
    1. Enter applicant details
    2. Get instant prediction
    3. Understand results with SHAP

    **Model Info**
    - Algorithm: XGBoost Classifier  
    - Tuned with RandomizedSearchCV  
    - Accuracy: ~81.92%  
    """)

elif pages == 'Data Info':
    st.header('📊 Data & Model Info')
    st.subheader('Quick Data Check:')
    st.dataframe(df.head(), use_container_width=True)

    st.subheader('Statistical Summary:')
    st.write(df.describe(), use_container_width=True)

    # Feature selection
    X = df[X_columns]
    Y = df['Loan_Status']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    st.subheader('Model Performance')
    st.write(f"✅ Accuracy: {accuracy_score(Y, y_pred)*100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(Y, y_pred))
    cm = confusion_matrix(Y, y_pred)
    st.write(f'Confusion Matrix: {cm}')

    # SHAP Plots
    st.subheader('Feature Importance')
    feature_names = X.columns
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_scaled, feature_names=X_columns, plot_type='bar', show=False)
    st.pyplot(fig)

    # Plotting Confusion Matrix
    st.subheader('Heatmap Presentation:')
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='d')
    st.pyplot(fig)
elif pages == 'Prediction Approval':
    st.header('📋 Loan Prediction Page')
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio('Gender', ['Male', 'Female'])
        married = st.selectbox('Married?', ['Yes', 'No'])
        dependents = st.slider('Dependents', 0, 10, 1)
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        self_emp = st.radio('Self Employed', ['Yes', 'No'])

    with col2:
        property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])
        credit_hist = st.selectbox('Credit History', ['Good (1)', 'Bad (0)'])
        app_income = st.number_input('Applicant Income', value=5000)
        coapp_income = st.number_input('Coapplicant Income', value=0)
        loan_amt = st.number_input('Loan Amount', value=100)
        loan_term = st.selectbox('Loan Term (months)', [360, 180, 120, 60])

    # Encoding inputs
    gender = 1 if gender == 'Male' else 0
    married = 1 if married == 'Yes' else 0
    education = 1 if education == 'Graduate' else 0
    self_emp = 1 if self_emp == 'Yes' else 0
    credit_hist = 1 if credit_hist.startswith('Good') else 0
    property_dict = {'Urban': 0, 'Rural': 1, 'Semiurban': 2}

    emi = loan_amt / (loan_term + 1e-5)
    total_inc = app_income + coapp_income
    inc_to_loan = total_inc / (loan_amt + 1e-5)

    sample = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_emp,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amt,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_hist,
        'Property_Area': property_dict[property_area],
        'Total_Income': total_inc,
        'EMI': emi,
        'Income_To_Loan': inc_to_loan
    }])

    if st.button('Predict Approval'):
        scaled_input = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
        prediction = model.predict(scaled_input)

        # Compute approval ratio
        approval_ratio = inc_to_loan  # Already engineered above

        if credit_hist == 0:
            st.error("❌ Not Qualified: Poor credit history")
        elif approval_ratio >= 50:
            approved_percentage = 1.0
            st.success("🎉 Approved: Full Loan (100%)")
        elif approval_ratio >= 30:
            approved_percentage = 0.75
            st.success("🎉 Approved: 75% of Loan")
        elif approval_ratio >= 20:
            approved_percentage = 0.5
            st.success("🎉 Approved: 50% of Loan")
        else:
            approved_percentage = 0.0
            st.error("❌ Not Qualified: Income too low vs loan amount")

        # ✅ Show actual approved amount (only if qualified)
        if approval_ratio >= 20 and credit_hist == 1:
            approved_amount = int(loan_amt * approved_percentage)
            st.info(f"**Approved Loan Amount: ${approved_amount}**")

        # SHAP explainability
        st.subheader("Explanation")
        explainer = shap.Explainer(model)
        shap_vals = explainer(scaled_input)
        feature_names = X_columns
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()



