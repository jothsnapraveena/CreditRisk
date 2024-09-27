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
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "Support Vector Machine": SVC(),
                "Naive Bayes": GaussianNB()
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models)

            # Find the best model based on test accuracy
            best_model_score = max(model_report.values(), key=lambda x: x['test_accuracy'])['test_accuracy']
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])

            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name}")
            logging.info(f"Best Model Test Accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)