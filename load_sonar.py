import pandas as pd
import numpy as np
import os,sys
import urllib
import csv
import warnings
warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/jaredvasquez/RandomForest/master/sonar.all-data.csv"

df = pd.read_csv(url, header=None)

## Make a directory to save the data

os.makedirs('Data', exist_ok=True)
df.to_csv('Data/sonar.csv')