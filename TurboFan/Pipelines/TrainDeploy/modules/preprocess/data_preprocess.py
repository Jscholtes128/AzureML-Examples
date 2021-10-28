import os
import argparse
import pandas as pd


# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--raw_data_dir', type=str, help='Directory where raw data is stored')
parser.add_argument('--train_dir', type=str, help='Directory to output the processed training data')
parser.add_argument('--test_dir', type=str, help='Directory to output the processed test data')
args = parser.parse_args()

# Get arguments from parser
raw_data_dir = args.raw_data_dir
train_dir = args.train_dir
test_dir = args.test_dir

# Make train, valid, test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

data = pd.read_csv(raw_data_dir + "turbofan.csv")
train_set = data.sample(frac=0.75, random_state=0)
test_set = data.drop(train_set.index)

train_set.to_csv(train_dir + "train.csv",index=False)
test_set.to_csv(test_dir + "test.csv",index=False)

    # Split into train, valid, test sets
    #num_images = len(image_files)
    #train_files = image_files[0:int(num_images*0.7)]
    #valid_files = image_files[int(num_images*0.7):int(num_images*0.9)]
    #test_files = image_files[int(num_images*0.9):num_images]

