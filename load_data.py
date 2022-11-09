
import pandas as pd
import numpy as np
import os,sys
from sklearn.datasets import load_iris
import csv

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.target

## Make a directory to save the data

os.makedirs('Data', exist_ok=True)
df.to_csv('Data/iris.csv')


