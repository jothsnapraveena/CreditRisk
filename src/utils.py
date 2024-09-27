import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder


from sklearn.exceptions import ConvergenceWarning
import warnings

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Use LabelEncoder to handle unexpected classes
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        logging.info(f"Unique classes in training set: {np.unique(y_train_encoded)}")
        logging.info(f"Unique classes in test set: {np.unique(y_test_encoded)}")

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            
            # Train the model
            model.fit(X_train, y_train_encoded)
            
            # Predict on train and test set
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
            test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
            precision = precision_score(y_test_encoded, y_test_pred, average='weighted',zero_division=0)
            recall = recall_score(y_test_encoded, y_test_pred, average='weighted',zero_division=0)
            f1 = f1_score(y_test_encoded, y_test_pred, average='weighted',zero_division=0)
            
            report[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logging.info(f"Model: {name}")
            logging.info(f"Train Accuracy: {train_accuracy}")
            logging.info(f"Test Accuracy: {test_accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1-score: {f1}")

        return report

    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)