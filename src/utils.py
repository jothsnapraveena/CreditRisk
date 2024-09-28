import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

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
    

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        # Use LabelEncoder to handle any categorical target variables
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        logging.info(f"Unique classes in training set: {np.unique(y_train_encoded)}")
        logging.info(f"Unique classes in test set: {np.unique(y_test_encoded)}")

        # Loop over models and perform hyperparameter tuning using GridSearchCV
        for name, model in models.items():
            logging.info(f"Tuning hyperparameters and training model: {name}")
            
            if name in params:
                # Perform hyperparameter tuning using GridSearchCV
                grid_search = GridSearchCV(estimator=model, param_grid=params[name], cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train_encoded)
                
                # Best model from grid search
                best_model = grid_search.best_estimator_
                logging.info(f"Best Params for {name}: {grid_search.best_params_}")
            else:
                # Train the model without hyperparameter tuning
                best_model = model
                best_model.fit(X_train, y_train_encoded)
            
            # Predict on training and test data
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
            test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
            precision = precision_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_test_pred, average='weighted', zero_division=0)
            
            # Store the results
            report[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'best_params': grid_search.best_params_ if name in params else None
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

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)