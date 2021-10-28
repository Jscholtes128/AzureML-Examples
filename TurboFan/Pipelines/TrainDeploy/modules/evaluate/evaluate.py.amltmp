from __future__ import print_function, division
import argparse
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.externals import joblib
import os


# Define arguments
parser = argparse.ArgumentParser(description='Evaluate arg parser')
parser.add_argument('--test_dir', type=str, help='Directory where testing data is stored')
parser.add_argument('--model_dir', type=str, help='Directory where model is stored')
parser.add_argument('--accuracy_file', type=str, help='File to output the accuracy to')
args = parser.parse_args()

# Get arguments from parser
test_dir = args.test_dir
model_dir = args.model_dir
accuracy_file = args.accuracy_file

# Load testing data, model, and device
test = pd.read_csv(test_dir + 'test.csv')
X_test = test.drop('rul',axis=1)
Y_test = pd.Series(test.rul)

model = joblib.load(model_dir + '/model.pkl')

y_pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, y_pred)


# Output accuracy to file
with open(accuracy_file, 'w+') as f:
    f.write(str(mae))