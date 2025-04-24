# import the libraries requried for data ingestion component

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging

@dataclass

class DataIngestionConfig:

    train_data_path:str = os.path.join('artificats', "train.csv")
    test_data_path:str = os.path.join('artificats', "test.csv")
    raw_data_path:str = os.path.join('artificats', "raw.csv")


#class for dataingestion

class DataIngestion:
    
     def __init__(self):
          self.ingestion_config = DataIngestionConfig()

     def initiate_config(self):
           logging.info("Entered the data ingestion method or component")
           try:
               prostate_df = pd.read_csv('notebook/data/Prostate_Cancer.csv')
               logging.info("Read the dataset into the prostate_df dataframe")

               os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
               prostate_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

               logging.info("Train_test split")
               train_set, test_set = train_test_split(prostate_df, test_size=0.2, random_state=42)
               train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
               test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

               logging.info("Data ingestion completed")
               return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
               )
           except Exception as e:
                raise CustomException
           

if __name__ == "__main__":
     
     obj = DataIngestion()
     obj.initiate_config()