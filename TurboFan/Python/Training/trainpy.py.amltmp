from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from random import randrange,randint
from azureml.core.run import Run
from sklearn.externals import joblib

max_depth = randint(2,10)
n_estimators = int(randrange(2000,5000,100))

train = pd.read_csv("data/turbofan.csv")

X = train.drop('rul',axis=1)
y = pd.Series(train.rul)

run = Run.get_context()

 # Log the algorithm parameter alpha to the run
run.log('max_depth', max_depth)
run.log('n_estimators', n_estimators)


X_train, X_test, y_train, y_test = train_test_split(X, y)


regression_model = GradientBoostingRegressor(
    max_depth=max_depth,
    n_estimators=n_estimators,
    learning_rate=.5
)

regression_model.fit(X_train, y_train)

y_pred = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
run.log('mae', mae)

# Save the model to the outputs directory for capture

#model_file_name = 'model_{}_{}.pkl'.format(max_depth,n_estimators)
model_file_name = 'model.pkl'
 

joblib.dump(value = regression_model, filename = model_file_name)

# upload the model file explicitly into artifacts 
run.upload_file(name = model_file_name, path_or_stream = model_file_name)

# Complete the run
run.complete()