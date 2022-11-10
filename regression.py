
import pandas as pd
import numpy as np
import os,sys
from sklearn.datasets import load_boston
import csv
import warnings
warnings.filterwarnings("ignore")

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['Price'] = boston.target


## Make a directory to save the data

os.makedirs('Data', exist_ok=True)
df.to_csv('Data/boston.csv')