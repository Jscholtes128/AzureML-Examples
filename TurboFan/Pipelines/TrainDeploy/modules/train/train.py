import argparse
import time
import os
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.externals import joblib


parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--train_dir', type=str, help='Directory where training data is stored')
parser.add_argument('--output_dir', type=str, help='Directory to output the model to')
parser.add_argument('--max_depth', type=int, help='max_depth')
parser.add_argument('--n_estimators', type=int, help='n_estimators')
args = parser.parse_args()

# Get arguments from parser
train_dir = args.train_dir
output_dir = args.output_dir
max_depth = args.max_depth
n_estimators = args.n_estimators

train = pd.read_csv(train_dir + 'train.csv')
X_train = train.drop('rul',axis=1)
Y_train = pd.Series(train.rul)


regression_model = GradientBoostingRegressor(
    max_depth=max_depth,
    n_estimators=n_estimators,
    learning_rate=.5
)

regression_model.fit(X_train, Y_train)


# Save model
print('Saving model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(value = regression_model, filename = output_dir + '/model.pkl' )



