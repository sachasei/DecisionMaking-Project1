import pandas as pd

data = pd.read_csv('dataset/RATP_csv.csv',sep=';', encoding='latin1')
print(data.head())

weights = [0.021,0.188,0.038,0.322,16.124,67.183,16.124]
