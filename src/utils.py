import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from src.exceptions import CustomException
from sklearn.model_selection import StratifiedKFold

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        results = {}
        for name, model in models.items():

        
        
            # fit on training data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]


            acc = accuracy_score(y_pred, y_test)
            auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)

            # record result

            results.append({
                "model" : name,
                "Accuracy": acc,
                "Precision" : report['1']['precision'],
                "Recall"   :  report['1']['recall'],
                'F1-Score' : report['1']['f1-score'],
                "ROC-AUC" : auc
            })
            results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False, ignore_index=True)
            return results_df
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)