import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sklearn_version
from imblearn.over_sampling import RandomOverSampler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'ApplicantIncome',
                'CoapplicantIncome',
                'LoanAmount',
                'Loan_Amount_Term',
                'Credit_History'
            ]
            categorical_columns = [
                'Gender',
                'Married',
                'Dependents',
                'Education',
                'Self_Employed',
                'Property_Area'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Check sklearn version to handle OneHotEncoder parameters
            if int(sklearn_version.split('.')[0]) >= 1:
                # For sklearn version 1.0 and above
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
                )
            else:
                # For sklearn versions below 1.0
                cat_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder", OneHotEncoder(drop='first', sparse=False)),
                        ("scaler", StandardScaler(with_mean=False))
                    ]
                )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Convert 'Loan_Status' to categorical
            train_df['Loan_Status'] = train_df['Loan_Status'].astype('category')
            test_df['Loan_Status'] = test_df['Loan_Status'].astype('category')

            # Convert categorical to numeric
            train_df['Loan_Status'] = train_df['Loan_Status'].cat.codes
            test_df['Loan_Status'] = test_df['Loan_Status'].cat.codes
            
            # Separate features and target
            X_train = train_df.drop('Loan_Status', axis=1)
            y_train = train_df['Loan_Status']

            # Apply random oversampling
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

            # Combine resampled features and target
            train_df = pd.concat([X_train_resampled, y_train_resampled], axis=1)
                

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head  : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Loan_Status'
            numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

            input_feature_train_df = train_df.drop(columns=[target_column_name, 'Loan_ID'], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, 'Loan_ID'], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.error("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

    print(f"Preprocessed train array shape: {train_arr.shape}")
    print(f"Preprocessed test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")