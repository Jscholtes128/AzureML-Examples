import argparse
import time
import os
from sklearn.metrics import average_precision_score,accuracy_score,f1_score,classification_report,precision_recall_curve,confusion_matrix,roc_auc_score,precision_score, recall_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer


parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--train_dir', type=str, help='Directory where training data is stored')
parser.add_argument('--output_dir', type=str, help='Directory to output the model to')
parser.add_argument('--n_estimators', type=int, help='n_estimators')
args = parser.parse_args()

# Get arguments from parser
train_dir = args.train_dir
output_dir = args.output_dir
n_estimators = args.n_estimators

train = pd.read_csv(train_dir + 'train.csv')
X_train = train.drop('Exited',axis=1)
Y_train = pd.Series(train.Exited)


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

rf = RandomForestClassifier(n_estimators=n_estimators)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('precprocessor', preprocessor),
    ('classifier', rf)
])

model.fit(X_train,Y_train)


# Save model
print('Saving model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

joblib.dump(value = model, filename = output_dir + '/model.pkl' )



