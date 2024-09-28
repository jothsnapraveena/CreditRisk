import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, features):
        try:
            logging.info("Starting prediction process")
            
            # Check if files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            logging.info("Loading model and preprocessor")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            logging.info("Model and preprocessor loaded successfully")
            
            logging.info("Transforming features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making prediction")
            preds = model.predict(data_scaled)
            
            logging.info("Prediction completed")
            return preds
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

# class PredictPipeline:
#     def __init__(self):
#         pass

#     def predict(self, features):
#         try:
#             model_path = os.path.join("artifacts", "model.pkl")
#             preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
#             print("Before Loading")
#             model = load_object(file_path=model_path)
#             preprocessor = load_object(file_path=preprocessor_path)
#             print("After Loading")
#             data_scaled = preprocessor.transform(features)
#             preds = model.predict(data_scaled)
#             return preds
        
#         except Exception as e:
#             raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Gender: str,
                 Married: str,
                 Dependents: str,
                 Education: str,
                 Self_Employed: str,
                 ApplicantIncome: float,
                 CoapplicantIncome: float,
                 LoanAmount: float,
                 Loan_Amount_Term: float,
                 Credit_History: float,
                 Property_Area: str):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
   
