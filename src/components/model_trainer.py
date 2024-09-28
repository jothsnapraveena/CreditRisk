import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Initializing model trainer...")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],
                                             train_array[:,-1], 
                                             test_array[:,:-1],
                                             test_array[:,-1]
            )

          
            # Convert y to integer type
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            logging.info(f"y_train dtype: {y_train.dtype}")
            logging.info(f"y_test dtype: {y_test.dtype}")
            logging.info(f"Unique values in y_train: {np.unique(y_train)}")
            logging.info(f"Unique values in y_test: {np.unique(y_test)}")

            models={
                "Logistic Regression":LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "Support Vector Machine": SVC()
               
            }

            params = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 5, 7]
                },
                "Support Vector Machine": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }


            logging.info("Starting model training with hyperparameter tuning")

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Find the best model based on test accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])
            best_model_score = model_report[best_model_name]['test_accuracy']

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Test Accuracy of the best model: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No model found with accuracy above threshold")

            logging.info("Saving the best model")

            # Fit the best model on the entire training data
            best_model.fit(X_train, y_train)


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Accuracy Score: {accuracy}")
            logging.info(f"Precision Score: {precision}")
            logging.info(f"Recall Score: {recall}")
            logging.info(f"F1 Score: {f1}")

            return {
                "best_model_name": best_model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }


        except Exception as e:
            raise CustomException(e, sys)




          