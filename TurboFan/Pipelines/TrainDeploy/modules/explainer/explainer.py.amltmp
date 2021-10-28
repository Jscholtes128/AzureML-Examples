from __future__ import print_function, division
import argparse
import time
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.externals import joblib
import os
from azureml.core.run import Run
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer


# Define arguments
parser = argparse.ArgumentParser(description='Evaluate arg parser')
parser.add_argument('--test_dir', type=str, help='Directory where testing data is stored')
parser.add_argument('--model_dir', type=str, help='Directory where model is stored')
parser.add_argument('--train_dir', type=str, help='File to output the accuracy to')
args = parser.parse_args()

# Get arguments from parser
test_dir = args.test_dir
model_dir = args.model_dir
train_dir = args.train_dir

# Load testing data, model, and device
test = pd.read_csv(test_dir + 'test.csv')
X_test = test.drop('rul',axis=1)
Y_test = pd.Series(test.rul)

train = pd.read_csv(train_dir + 'train.csv')
X_train = train.drop('rul',axis=1)

model = joblib.load(model_dir + '/model.pkl')

run = Run.get_context()
client = ExplanationClient.from_run(run)
explainer = TabularExplainer(model, 
                             X_train)

# explain overall model predictions (global explanation)
global_explanation = explainer.explain_global(X_test)

# uploading global model explanation data for storage or visualization in webUX
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, comment='global explanation: all features')

