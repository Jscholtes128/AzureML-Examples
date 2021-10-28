import os
import requests
import argparse
import numpy as np
import pandas as pd

# import needed libraries for downloading and unzipping the file
import urllib.request
from zipfile import ZipFile

parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--output_dir', type=str, help='Directory to store output raw data')

args = parser.parse_args()

# Get arguments from parser
output_dir = args.output_dir

#Download file
response = urllib.request.urlopen("https://ti.arc.nasa.gov/c/6/")
output = open(output_dir + 'CMAPSSData.zip', 'wb')    # note the flag:  "wb"        
output.write(response.read())
output.close()


# unzip files
zipfile = ZipFile(output_dir + "CMAPSSData.zip")
zipfile.extract("train_FD001.txt", path=output_dir + "/data/")

train = pd.read_csv(output_dir + "/data/train_FD001.txt", delimiter="\s|\s\s", index_col=False, engine='python', names=['unit','cycle','os1','os2','os3','sm1','sm2','sm3','sm4','sm5','sm6','sm7','sm8','sm9','sm10','sm11','sm12','sm13','sm14','sm15','sm16','sm17','sm18','sm19','sm20','sm21'])
# operational settings and sensor measurements

# Our dataset has a number of units in it, with each engine flight listed as a cycle.
# The cycles count up until the engine fails. What we would like to predict is the number of cycles until failure.
# So we need to calculate a new column called RUL, or Remaining Useful Life. 
#It will be the last cycle value minus each cycle value per unit.
def assignrul(df):
    maxi = df['cycle'].max()
    df['rul'] = maxi - df['cycle']
    return df
    

train_new = train.groupby('unit').apply(assignrul)

# Display Columns
print(train_new.columns)

train_new.to_csv(output_dir + 'turbofan.csv', index=False)
