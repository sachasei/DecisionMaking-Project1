import pandas as pd
import problem_resolution as pr

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
score_final_all = data.sort_values(by='Score_Final', ascending=False)
score_final_all.to_csv('dataset/RATP_score_final.csv', index=False, sep=';')



# 2. Liste des colonnes de critères (noms originaux)
criteria_cols = [
    'peak-entering-passengers/h', 
    'peak-passing-passengers/h', 
    'off-peak-entering-passengers/h', 
    'off-peak-passing-passengers/h', 
    'strategic priority [0,10]', 
    'Station degradation level ([0,20]  scale)', 
    'connectivity index [0,100]'
]

# 3. Extraction de la première ligne comme référence (x)
first_row = score_final_all.iloc[0]

# 4. Construction du dictionnaire des différences (x - y)
diff_dict = {}

for i in range(1, len(score_final_all)):
    current_row = score_final_all.iloc[i]
    station_name = current_row['Metro station']
    
    # Calcul des différences : Ligne_1 - Ligne_i
    row_diffs = {}
    for col in criteria_cols:
        val_diff = first_row[col] - current_row[col]
        # On force en float pour la compatibilité JSON et on arrondit
        row_diffs[col] = float(round(val_diff, 3))
    
    diff_dict[station_name] = row_diffs

# 5. Affichage d'un exemple pour vérifier
#print(diff_dict)


for entry in diff_dict.keys():
    print(f"Trade-offs for station: {entry}")
    pros,cons,neutral = pr.preprocess_data(diff_dict[entry])
    tradeoffs = pr.m_to_one_or_one_to_m_tradeoffs(diff_dict[entry], pros, cons)
    print(tradeoffs)
    print("\n")
