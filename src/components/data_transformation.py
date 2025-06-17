import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exceptions import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_file =os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_datatransformer(self):

        try:
            feature_column = [
                'radius', 'texture', 'perimeter', 'area',
                'smoothness', 'compactness', 'symmetry', 'fractal_dimension'
             ]
            
            feature_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Feature columns: {feature_column}")
            preprocessor =ColumnTransformer(
                ["feature_pipeline", feature_pipeline, feature_column]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining proprocessor")
            preprocessor_file_obj = self.get_datatransformer()
            target_name = "diagnosis_result"
            feature_column = ['radius', 'texture', 'perimeter', 'area',
                'smoothness', 'compactness', 'symmetry', 'fractal_dimension']
            X_feature_train_df = train_df.drop(columns=[target_name], axis=1)
            y_target_train_df = train_df[target_name]
            X_feature_test_df =test_df.drop(columns=[target_name], axis=1)
            y_target_test_df = test_df[target_name]
            logging.info("Applying preprocessing to dataframe")
            feature_train_arr = preprocessor_file_obj.fit_transform(X_feature_train_df)
            feature_test_arr = preprocessor_file_obj.transform(X_feature_test_df)
            train_arr = np.c_[
                feature_train_arr, np.array(X_feature_train_df)
            ]
            test_arr = np.c_[
                feature_test_arr, np.array(X_feature_test_df)
            ]
            logging.info(f"Saved Processing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_file, obj=preprocessor_file_obj)
        
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_file, 
            )
    
        except Exception as e:  
            raise CustomException(e, sys)
        


