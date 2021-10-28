
# Change the name based on the randomly generated filename
# Scoring Script will need model id from registered model
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from azureml.core.model import Model
import os
from azureml.monitoring import ModelDataCollector
import time

def init():
    global model
    global inputs_dc, prediction_dc

    inputs_dc = ModelDataCollector("best_model", designation="inputs", feature_names=["unit", "cycle", "os1", "os2", "os3", "sm1","sm2","sm3","sm4","sm5","sm6","sm7","sm8","sm9","sm10","sm11","sm12","sm13","sm14","sm15","sm16","sm17","sm18","sm19","sm20","sm21"])
    prediction_dc = ModelDataCollector("best_model", designation="predictions", feature_names=["prediction"])
    print("model initialized" + time.strftime("%H:%M:%S"))
    # retreive the path to the model file using the model name
    pth = Model.get_model_path('turbofan-pipeline-rul')
    model_path = pth 
    #os.path.join(pth, 'model.pkl')
    #os.getenv('AZUREML_MODEL_DIR')
    model = joblib.load(model_path)
    

def run(raw_data):
    # grab and prepare the data
    data = pd.read_json(json.loads(raw_data),orient='records')

    y_hat = model.predict(data)
    print("Prediction created" + time.strftime("%H:%M:%S"))

    inputs_dc.collect(data) 
    prediction_dc.collect(y_hat)
    return json.dumps(y_hat.tolist())
    return json.dumps(y_hat.tolist())