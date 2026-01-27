import pandas as pd
import problem_resolution as pr

data = pd.read_csv('dataset/data27crit.csv',sep=';', encoding='latin1')

weights = [2,1,5,3,4,6,1,7,4,3,8,5,1,2,1,4,5,6,2,7,1,2,6,3,5,5,1]

# Conversion des colonnes numériques (remplacer les virgules par des points)
numeric_columns = data.columns[1:]  # Toutes les colonnes sauf 'solution name'
for col in numeric_columns:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Calcul de la somme pondérée
data['Score_Final'] = 0

# On multiplie chaque colonne par son poids respectif
for col, weight in zip(numeric_columns, weights):
    data['Score_Final'] += data[col] * weight

# Tri par score décroissant
data_sorted = data.sort_values(by='Score_Final', ascending=False)

# Affichage des scores dans les logs
print("\n=== Scores des solutions (triés par score décroissant) ===")
for idx, row in data_sorted.iterrows():
    print(f"{row['solution name']}: {row['Score_Final']:.2f}")

# Extraction de la première ligne comme référence (meilleure solution)
first_row = data_sorted.iloc[0]

# Construction du dictionnaire des différences (meilleure solution - solution courante)
diff_dict = {}

for i in range(1, len(data_sorted)):
    current_row = data_sorted.iloc[i]
    solution_name = current_row['solution name']
    
    # Calcul des différences : Meilleure solution - Solution_i
    row_diffs = {}
    for col in numeric_columns:
        val_diff = first_row[col] - current_row[col]
        # On force en float pour la compatibilité JSON et on arrondit
        row_diffs[col] = float(round(val_diff, 3))
    
    diff_dict[solution_name] = row_diffs

# Calcul des trade-offs pour chaque comparaison
print("\n=== Trade-offs : Meilleure solution vs autres solutions ===")
for entry in diff_dict.keys():
    print(f"\nTrade-offs for solution: {entry}")
    pros, cons, neutral = pr.preprocess_data(diff_dict[entry])
    tradeoffs = pr.m_to_one_or_one_to_m_tradeoffs(diff_dict[entry], pros, cons)
    print(tradeoffs)

# Extraction de la deuxième ligne comme référence (deuxième meilleure solution)
second_row = data_sorted.iloc[1]

# Construction du dictionnaire des différences (deuxième meilleure solution - solution courante)
# On exclut la meilleure solution (index 0)
diff_dict_second = {}

for i in range(2, len(data_sorted)):
    current_row = data_sorted.iloc[i]
    solution_name = current_row['solution name']
    
    # Calcul des différences : Deuxième meilleure solution - Solution_i
    row_diffs = {}
    for col in numeric_columns:
        val_diff = second_row[col] - current_row[col]
        # On force en float pour la compatibilité JSON et on arrondit
        row_diffs[col] = float(round(val_diff, 3))
    
    diff_dict_second[solution_name] = row_diffs

# Calcul des trade-offs pour chaque comparaison
print("\n=== Trade-offs : Deuxième meilleure solution vs autres solutions (sans la meilleure) ===")
for entry in diff_dict_second.keys():
    print(f"\nTrade-offs for solution: {entry}")
    pros, cons, neutral = pr.preprocess_data(diff_dict_second[entry])
    tradeoffs = pr.m_to_one_or_one_to_m_tradeoffs(diff_dict_second[entry], pros, cons)
    print(tradeoffs)

# Extraction de la troisième ligne comme référence (troisième meilleure solution)
third_row = data_sorted.iloc[2]

# Construction du dictionnaire des différences (troisième meilleure solution - solution courante)
# On exclut la meilleure solution (index 0) et la deuxième meilleure (index 1)
diff_dict_third = {}

for i in range(3, len(data_sorted)):
    current_row = data_sorted.iloc[i]
    solution_name = current_row['solution name']
    
    # Calcul des différences : Troisième meilleure solution - Solution_i
    row_diffs = {}
    for col in numeric_columns:
        val_diff = third_row[col] - current_row[col]
        # On force en float pour la compatibilité JSON et on arrondit
        row_diffs[col] = float(round(val_diff, 3))
    
    diff_dict_third[solution_name] = row_diffs

# Calcul des trade-offs pour chaque comparaison
print("\n=== Trade-offs : Troisième meilleure solution vs autres solutions (sans la première et la deuxième) ===")
for entry in diff_dict_third.keys():
    print(f"\nTrade-offs for solution: {entry}")
    pros, cons, neutral = pr.preprocess_data(diff_dict_third[entry])
    tradeoffs = pr.m_to_one_or_one_to_m_tradeoffs(diff_dict_third[entry], pros, cons)
    print(tradeoffs)
