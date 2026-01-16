import pandas as pd

data = pd.read_csv('dataset/RATP_csv.csv',sep=';', encoding='latin1')
print(data.head())

weights = [0.021,0.188,0.038,0.322,16.124,67.183,16.124]


#3. Sélection des colonnes numériques
# On prend toutes les colonnes à partir de la deuxième (on exclut 'Metro station')
numeric_columns = data.columns[1:]

# 4. Calcul de la somme pondérée
# On initialise la nouvelle colonne à 0
data['Score_Final'] = 0

# On multiplie chaque colonne par son poids respectif
for col, weight in zip(numeric_columns, weights):
    data['Score_Final'] += data[col] * weight

# 5. Affichage et sauvegarde
score_final=data[['Metro station', 'Score_Final']].sort_values(by='Score_Final', ascending=False)

print(score_final)
