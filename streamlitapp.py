import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")
    
    st.title("Loan Approval Predictor")
    st.header("Loan Application Status Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=0)
        coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=0)
        loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=0)
        credit_history = st.selectbox("Credit History", ["1", "0"], format_func=lambda x: "Good (1)" if x == "1" else "Bad (0)")

    if st.button("Predict Loan Status"):
        data = CustomData(
            Gender=gender,
            Married=married,
            Dependents=dependents,
            Education=education,
            Self_Employed=self_employed,
            ApplicantIncome=float(applicant_income),
            CoapplicantIncome=float(coapplicant_income),
            LoanAmount=float(loan_amount),
            Loan_Amount_Term=float(loan_amount_term),
            Credit_History=float(credit_history),
            Property_Area=property_area
        )
        
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        if results[0] == 1:
            st.success("Congratulations! The loan is likely to be approved.")
        else:
            st.error("Unfortunately, the loan is likely to be denied.")

        st.subheader("Prediction Details")
        st.write(pd.DataFrame(pred_df))

if __name__ == "__main__":
    main()