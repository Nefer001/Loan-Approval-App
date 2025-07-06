import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

st.set_page_config(page_title='Loan Approval Predictor', layout='wide')

@st.cache_data
#load data
def load_data():
    df = pd.read_csv('Loan Prediction.csv')

    # Fill Nulls
    num_imputer = SimpleImputer(strategy='median')
    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Encode Categorical Data
    encoder = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    # Clean Encoding
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype('float')

    # Advance Engineering the features
    df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-6)
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_To_Loan'] = df['Total_Income'] / (df['LoanAmount'] + 1e-6)

    return df, encoder
df, encoder = load_data()

#Train And Cache data
@st.cache_resource
def train_model():
    #feature selection
    X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area',
            'Total_Income', 'EMI', 'Income_To_Loan']]
    Y = df['Loan_Status']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Apply scale_pos_weight to balance classes
    pos_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=pos_weight, random_state=42)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # Apply RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, scoring='accuracy',
                                       cv=5, verbose=1, random_state=42, n_jobs=1)
    random_search.fit(X_train_scaled, Y_train)

    return random_search.best_estimator_, scaler, X_test_scaled, Y_test, X
model, scaler, X_test_scaled, Y_test, X = train_model()

#Navigation
st.sidebar.header('🧭 Navigation Pages')
pages = st.sidebar.radio('Select Page', ['Home', 'Data Info', 'Prediction Approval'])

if pages == 'Home':
    st.title('Loan Approval Predictor 💰')
    st.markdown('''Welcome to The Loan Approval Prediction App!\n  This Tool Uses a Powerful AI Model(XGBoost) To Predict Whether A Loan Application Is Likely To Be Approved or Not Approved\n Based on [real-world financial dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle.

    What You can Do:
    1.Enter details about the Loan Applicant
    2.Get Instant Prediction results
    3.Understand the reason Using SHAP Explainability

    Model Info
    1.Model:XGBoost Classifier
    2.Tuned:RandomizedSearchCV
    3.Accuracy: 83% 
    ---
    ''')

elif pages == 'Data Info':
    st.header('Prediction Result')
    st.subheader('Quick Data Review')
    st.dataframe(df.head(), use_container_width=True)

    st.subheader('Statistical Report:')
    st.write(df.describe(), use_container_width=True)

    st.subheader('Model Performance')
    y_predict = model.predict(X_test_scaled)

    st.write(f'Accuracy Score: {accuracy_score(Y_test, y_predict)*100:.2f}%')
    st.write(f'Classification Report:')
    st.text(classification_report(Y_test, y_predict))

    #SHAP explainability
    st.subheader('Feature Importance')
    feature_names = X.columns
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_scaled)

    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type='bar')
    st.pyplot(fig)

    st.subheader('Feature impact on Prediction')
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 2, 3])
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
    st.pyplot(fig)

    #Load Full Dataset
    st.subheader('Full Dataset')
    st.dataframe(df, use_container_width=True)

elif pages == 'Prediction Approval':
    st.header('Loan Prediction Page')

    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio('Select Gender', ['Male', 'Female'])
        marry_status = st.selectbox('Married status', ['Yes', 'No'])
        dependents = st.slider('Number Of Dependencies', 0, 10, 3)
        edu = st.selectbox('Education Status', ['Graduate', 'Not Graduate'])
        employment = st.radio('Self Employed', ['Yes', 'No'])

    with col2:
        property = st.selectbox('Home Location', ['Urban', 'Rural', 'Semiurban'])
        cred_hst = st.selectbox('Credit History', ['Good(1)', 'Bad(0)'])
        applicant_inc = st.number_input('Applicant Income (₹)', min_value=0, value=0)
        coapplicant_inc = st.number_input('Coapplicant Income (₹)', min_value=0, value=0)
        loan_amnt = st.number_input('Loan Amount (₹)', min_value=0, max_value=1000000)
        loan_term = st.selectbox('Loan Amount Term (months)', [360, 180, 60, 120])

    # Engineering Inputs
    gender = 1 if gender == 'Male' else 0
    marry_status = 1 if marry_status == 'Yes' else 0
    edu = 1 if edu == 'Graduate' else 0
    employment = 1 if employment == 'Yes' else 0
    property_nav = {'Urban': 0, 'Rural': 1, 'Semiurban': 2}
    cred_hst = 1 if cred_hst.startswith('Good') == 1 else 0

    # Advance Engineering the features
    emi = loan_amnt / (loan_term + 1e-5) #Equated Monthly Installment -> Fixed Monthly amount a borrower must pay to repay a loan(including interest
    total_income = applicant_inc + coapplicant_inc
    income_to_loan = total_income / (loan_amnt + 1e-5)


    # ✅ FIXED SAMPLES: Keep only features used in training
    samples = pd.DataFrame({
        'Gender': [gender],
        'Married': [marry_status],
        'Dependents': [dependents],
        'Education': [edu],
        'Self_Employed': [employment],
        'ApplicantIncome': [applicant_inc],
        'CoapplicantIncome': [coapplicant_inc],
        'LoanAmount': [loan_amnt],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [cred_hst],
        'Property_Area': [property_nav[property]],
        'Total_Income': [total_income],
        'EMI': [emi],
        'Income_To_Loan': [income_to_loan]
    })

    if st.button('Predict Loan Approval'):
        input_scaled = scaler.transform(samples)
        prediction = (model.predict_proba(input_scaled)[:, 1] >= 0.45)  # Lower threshold
        proba = model.predict_proba(input_scaled)
        #Finally Prediction On The results
        approval_ratio = income_to_loan
        if cred_hst == 0:
            st.error('❌ Not Qualified, Poor Credit History')
        elif approval_ratio >= 50:
            approved_percentage = 0.1
            st.success('🎉Approved: You Get 100% Loan')
        elif approval_ratio >=30:
            approved_percentage = 0.75
            st.success('🎉Approved: You Get 75% of Loan')
        elif approval_ratio >=20:
            approved_percentage = 0.5
            st.success('🎉Approved: You Get 50% of Loan')
        else:
            approved_percentage = 0.0
            st.error('❌ Not Qualified to get Loan, Your Income Is Low Against Loan Amount')

        #Displating Amount Obtained
        if approval_ratio >= 20 and cred_hst==1:
            approved_amount = int(loan_amnt * approved_percentage)
            st.info(f'Approved Loan Amount: {approved_amount}')

        # SHAP explainability
        st.subheader('Prediction Explanation')
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)
        feature_names = X.columns

        shap.summary_plot(shap_values,
                          input_scaled,
                          feature_names=feature_names,
                          plot_type='dot',
                          show=False)

        plt.title('Feature Impact on Decision', fontsize=12)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()