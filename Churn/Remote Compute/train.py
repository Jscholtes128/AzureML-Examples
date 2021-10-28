import argparse
import time
import os
from sklearn.metrics import average_precision_score,accuracy_score,roc_auc_score,precision_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from azureml.core import Dataset, Run
from sklearn.model_selection import train_test_split

# Get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=10)
parser.add_argument('--max_depth', type=int, dest='max_depth', default=2)
args = parser.parse_args()
n_estimators = args.n_estimators
max_depth = args.max_depth


run = Run.get_context()



ws = run.experiment.workspace
dataset = Dataset.get_by_name(ws, name='Churn Databricks')
train = dataset.to_pandas_dataframe()

X_train = train.drop('Exited',axis=1)
Y_train = pd.Series(train.Exited)

X = train.drop('Exited', axis=1)
y = train['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)

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

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
run.log('Accuracy', np.float(accuracy))



os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')



