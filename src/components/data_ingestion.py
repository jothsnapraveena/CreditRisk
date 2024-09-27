import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformationConfig

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig( )
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_csv(r'D:\End To End ML\notebook\CreditRisk.csv')

            # Ensure 'Loan_Status' is of category type
            df['Loan_Status'] = df['Loan_Status'].astype(str)

            logging.info("Data ingestion Initiated")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            logging.info("Train Test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    try:
        obj=DataIngestion()
        train_data,test_data=obj.initiate_data_ingestion()
        print(f"Data ingestion completed. Train data: {train_data}, Test data: {test_data}")

        data_transformation=DataTransformation()
        train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data,test_data)
        print(f"Data transformation completed. Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")

        modeltrainer=ModelTrainer()
        result = modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path)
        print(f"Model training completed. Result: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")
