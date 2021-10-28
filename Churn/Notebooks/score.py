
# Change the name based on the randomly generated filename
# Scoring Script will need model id from registered model
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from azureml.core.model import Model
import os
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer

def init():
    global model
 
    print("model initialized" + time.strftime("%H:%M:%S"))
    # retreive the path to the model file using the model name
    pth = Model.get_model_path('churn-classifier')
    model_path = pth 
    model = joblib.load(model_path)
    
def run(raw_data):
    # grab and prepare the data
    data = pd.read_json(json.loads(raw_data),orient='records')

    y_hat = model.predict(data)
    print("Prediction created" + time.strftime("%H:%M:%S"))

    return json.dumps(y_hat.tolist())