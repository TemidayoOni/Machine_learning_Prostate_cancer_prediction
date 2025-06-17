import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from src.exceptions import CustomException
from src.logger import logging
from sklearn.model_selection import StratifiedKFold
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl" )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split the data int train and test")
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:, -1]

            )
            models = {
  
                'Logistic Regression' : LogisticRegression(max_iter=1000),
                'Random Forest' : RandomForestClassifier(n_estimators=100),
                'Support Vector Machine' : SVC(probability=True),
                'KNN': KNeighborsClassifier(),
                'Neural Net' : MLPClassifier(max_iter=1000),
            }
            

            # cross-validation setup

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            result:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test, models=models)
            best_model_score = max(sorted(result.values()))

            

            best_model_name = list(result.keys())[
                list(result.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc = accuracy_score(y_test, predicted)
            return acc
            



            
        except Exception as e:
            raise CustomException(e,sys)