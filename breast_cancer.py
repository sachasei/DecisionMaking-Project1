import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import problem_resolution as pr

# Configurer l'affichage de pandas pour plus de décimales
pd.set_option('display.precision', 6)
pd.set_option('display.float_format', '{:.6f}'.format)

# Charger les données
df = pd.read_csv('dataset/breastcancer_processed.csv')

# Afficher les premières lignes et informations du dataset
print("Aperçu des données:")
print(df.head())
print("\nInformations du dataset:")
print(df.info())
print("\nStatistiques descriptives:")
print(df.describe())
print("\nDistribution de la variable cible 'Benign':")
print(df['Benign'].value_counts())

# Séparer les features (X) et la cible (y)
X = df.drop('Benign', axis=1)
y = df['Benign']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTaille de l'ensemble d'entraînement: {X_train.shape[0]}")
print(f"Taille de l'ensemble de test: {X_test.shape[0]}")

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle de régression logistique
print("\n" + "="*50)
print("Entraînement du modèle de régression logistique...")
print("="*50)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Faire des prédictions
y_pred_train = logistic_model.predict(X_train_scaled)
y_pred_test = logistic_model.predict(X_test_scaled)

# Évaluer le modèle
print("\n" + "="*50)
print("RÉSULTATS DU MODÈLE")
print("="*50)

print("\nPrécision sur l'ensemble d'entraînement:", accuracy_score(y_train, y_pred_train))
print("Précision sur l'ensemble de test:", accuracy_score(y_test, y_pred_test))

print("\nRapport de classification (Ensemble de test):")
print(classification_report(y_test, y_pred_test, target_names=['Malignant (0)', 'Benign (1)']))

# Matrice de confusion
print("\nMatrice de confusion:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Visualiser la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title('Matrice de Confusion - Régression Logistique')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.tight_layout()
plt.savefig('confusion_matrix_breast_cancer.png')
print("\nMatrice de confusion sauvegardée sous 'confusion_matrix_breast_cancer.png'")

# Afficher les coefficients du modèle
print("\n" + "="*50)
print("COEFFICIENTS DU MODÈLE")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logistic_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nImportance des features (coefficients):")
print(feature_importance)

# Visualiser les coefficients
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Coefficients de la Régression Logistique')
plt.tight_layout()
plt.savefig('feature_coefficients_breast_cancer.png')
print("\nCoefficients sauvegardés sous 'feature_coefficients_breast_cancer.png'")

# Calculer le score Benign pour toutes les données
print("\n" + "="*50)
print("CALCUL DES SCORES BENIGN")
print("="*50)
X_scaled = scaler.transform(X)
coefficients = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]

# Produit scalaire : coefficients @ x_scaled + intercept
df['Benign_Score'] = X_scaled @ coefficients + intercept

# Calculer la probabilité avec la fonction sigmoïde
df['Benign_Prob'] = logistic_model.predict_proba(X_scaled)[:, 1]

print("\nAperçu des scores Benign calculés:")
print(df[['Benign', 'Benign_Score', 'Benign_Prob']].head(10))
print(f"\nStatistiques des scores:")
print(df['Benign_Score'].describe())
print(f"\nStatistiques des probabilités:")
print(df['Benign_Prob'].describe())


df.sort_values(by='Benign_Score', ascending=False, inplace=True)

print("\nTop 5 des échantillons avec les scores Benign les plus élevés:")
print(df[['Benign', 'Benign_Score', 'Benign_Prob']].head(5))


def diff_dict(df, y1, y2):
    row1 = df.loc[y1]
    row2 = df.loc[y2]
    
    features = [col for col in df.columns if col not in ['Benign', 'Benign_Score', 'Benign_Prob']]
    diffs = {}
    for feature in features:
        weight = feature_importance[feature_importance['Feature'] == feature]['Coefficient'].values[0]
        diffs[feature] = weight*(row1[feature] - row2[feature])
    return diffs
     
     

def explain(df, y1, y2):
    """
    Explique pourquoi y1 > y2 ou inversement, en utilisant les trade_offs.
    """
    row1 = df.loc[y1]
    row2 = df.loc[y2]
    if row1['Benign_Score'] == row2['Benign_Score']:
        return f"Les échantillons {y1} et {y2} ont le même score Benign de {row1['Benign_Score']:.6f}."
    elif row1['Benign_Score'] > row2['Benign_Score']:
        better, worse = y1, y2
    else:
        better, worse = y2, y1
    
    data = diff_dict(df, better, worse)
    pros,cons,neutral = pr.preprocess_data(data)
    results = pr.m_to_one_or_one_to_m_tradeoffs(data, pros, cons)
    if type(results) == str:
       return f'Aucun trade-off identifié entre les échantillons {y1} et {y2}.'
    else:
        return pros, cons, neutral, data, better, worse, results
    


def explanation(df,y1,y2):
    explanation = explain(df, y1, y2)
    print(f'\nComparaison entre les échantillons {y1} et {y2}:')
    if type(explanation) == str:
        print(f'\n{explanation}')
    else:
        pros, cons, neutral, data, better, worse, results = explanation
        print(f'\nAnalyse des différences entre les échantillons {y1} et {y2}:')
        print(f'\nAspects positifs (pros) pour l\'échantillon {better} avec le score le plus élevé:')
        for p in pros:
            print(f' - {p}: {data[p]:.6f}')
        print(f'\nAspects négatifs (cons) pour l\'échantillon {better} avec le score le plus élevé:')
        for c in cons:
            print(f' - {c}: {data[c]:.6f}')
        print(f'\nAspects neutres:')
        for n in neutral:
            print(f' - {n}: {data[n]:.6f}')
        print(f'\nTrade-offs identifiés:')
        if results == []:
            print('Aucun contre-argument, pas besoin de trade-off.')
        else:
            print(results)

#Exemple avec le meilleur échantillon 
explanation(df, 597, 346)
# Quelques exemples aléatoires:
import random
random_indices = random.sample(list(df.index), 4)
for i in range(0, 4, 2):
    y1 = random_indices[i]
    y2 = random_indices[i+1]
    explanation(df, y1, y2)
    



# Exemple entre les 2 meilleurs échantillons: aucune explication possible.
#C'est du au fait que les coefficients sont appris sur des données scalées, et que les scores sont très proches.
explanation(df, 597, 277)   
data = diff_dict(df, 597, 277)
pros, cons, neutral = pr.preprocess_data(data)
print(f'\nAspects positifs (pros) pour l\'échantillon 597:')
for p in pros:
    print(f' - {p}: {data[p]:.6f}')
print(f'\nAspects négatifs (cons) pour l\'échantillon 597:')
for c in cons:
    print(f' - {c}: {data[c]:.6f}')     

 