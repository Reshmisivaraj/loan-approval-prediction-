import pickle
import streamlit as st

try:
    with open('LoanAmountPrediction.pkl', 'rb') as f:
        model = pickle.load(f)
    st.write('Model loaded successfully')
except Exception as e:
    st.error(f'Error loading model: {e}')

def loan_status_prediction(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, threshold=0.85):
    input_data = [[no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]]
    predicted_loan_status = model.predict(input_data)[0]
    return 1 if predicted_loan_status > threshold else 0

def main():
    st.title('Loan Sanction Prediction')
    threshold = 0.9 
    no_of_dependents = st.number_input('Enter Number of Dependents:', min_value=0, step=1)
    education = st.number_input('Enter Education (1: Graduate, 0: Not Graduate):', min_value=0, max_value=1, step=1)
    self_employed = st.number_input('Self Employed (1: Yes, 0: No):', min_value=0, max_value=1, step=1)
    income_annum = st.number_input('Enter Annual Income:', min_value=0, step=1000)
    loan_amount = st.number_input('Enter Loan Amount:', min_value=0, step=1000)
    loan_term = st.number_input('Enter Loan Term:', min_value=0, step=1)
    cibil_score = st.number_input('Enter CIBIL Score:', min_value=0, step=1)
    residential_assets_value = st.number_input('Enter Residential Assets Value:', min_value=0, step=1000)
    commercial_assets_value = st.number_input('Enter Commercial Assets Value:', min_value=0, step=1000)
    luxury_assets_value = st.number_input('Enter Luxury Assets Value:', min_value=0, step=1000)
    bank_asset_value = st.number_input('Enter Bank Assets Value:', min_value=0, step=1000)

    if st.button('Predict Loan Sanction'):
        predicted_status = loan_status_prediction(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, threshold)
        
        if predicted_status == 1:
            st.success('Loan will be sanctioned.')
        else:
            st.error('Loan will not be sanctioned.')

if __name__ == "__main__":
    main()
