import os
import argparse
import pandas as pd

from azureml.core import Dataset, Run


# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--train_dir', type=str, help='Directory to output the processed training data')
parser.add_argument('--test_dir', type=str, help='Directory to output the processed test data')

args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

dataset_name = 'Churn Databricks'

dataset = Dataset.get_by_name(ws, name=dataset_name)
data = dataset.to_pandas_dataframe()


# Get arguments from parser
train_dir = args.train_dir
test_dir = args.test_dir

# Make train, valid, test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_set = data.sample(frac=0.75, random_state=0)
test_set = data.drop(train_set.index)

train_set.to_csv(train_dir + "train.csv",index=False)
test_set.to_csv(test_dir + "test.csv",index=False)

    # Split into train, valid, test sets
    #num_images = len(image_files)
    #train_files = image_files[0:int(num_images*0.7)]
    #valid_files = image_files[int(num_images*0.7):int(num_images*0.9)]
    #test_files = image_files[int(num_images*0.9):num_images]

