import pandas as pd

data = pd.read_csv('dataset/RATP_csv.csv',sep=';', encoding='latin1')
print(data.head())


