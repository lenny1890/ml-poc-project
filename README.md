# SNCF TGV Delay Predictor

Proof of Concept de Machine Learning pour predire le retard moyen a l'arrivee des TGV en France, a partir des donnees Open Data SNCF.

## Objectif

Predire le **retard moyen a l'arrivee** (en minutes) d'une liaison TGV donnee, pour un mois donne, en exploitant les donnees historiques de regularite publiees par la SNCF.

## Dataset

- **Source** : [SNCF Open Data - Regularite mensuelle TGV par liaisons](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/)
- **Volume** : 12 181 observations (2018-2026)
- **Granularite** : mensuelle, par liaison gare a gare
- **Variable cible** : Retard moyen de tous les trains a l'arrivee (minutes)
- **Features** : gare de depart, gare d'arrivee, duree du trajet, nombre de circulations prevues, nombre de trains annules, annee, mois

Le fichier CSV n'est pas versionne (trop volumineux). Pour reproduire le projet, telechargez-le depuis le lien ci-dessus et placez-le dans `data/sncf_retards.csv`.

## Resultats

| Modele | MAE (min) | RMSE (min) | R2 |
|--------|-----------|------------|------|
| Gradient Boosting | 1.89 | 2.74 | 0.50 |
| Random Forest | 1.89 | 2.77 | 0.49 |
| K-Nearest Neighbors | 2.17 | 3.06 | 0.38 |
| Linear Regression | 2.66 | 3.59 | 0.15 |

Le **Gradient Boosting** est le modele retenu, avec une erreur moyenne de 1.9 minutes et un R2 de 0.50.

## Structure du projet

    ml-poc-project/
    ├── data/                  # Donnees brutes (non versionnees)
    ├── models/                # Modeles entraines (.joblib)
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   └── 02_preprocessing_training.ipynb
    ├── plots/
    ├── results/               # model_metrics.csv
    ├── scripts/
    │   └── main.py
    ├── src/
    │   ├── config.py
    │   ├── data.py
    │   ├── metrics.py
    │   └── app.py
    ├── tests/
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Installation et execution

### 1. Cloner le depot

    git clone git@github.com:lenny1890/ml-poc-project.git
    cd ml-poc-project

### 2. Creer l'environnement virtuel

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 3. Telecharger les donnees

Telechargez le CSV depuis [SNCF Open Data](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/) et placez-le dans `data/sncf_retards.csv`.

### 4. Entrainer les modeles

Executez les notebooks dans l'ordre :

    notebooks/01_eda.ipynb
    notebooks/02_preprocessing_training.ipynb

### 5. Lancer le pipeline complet

    python scripts/main.py

Cette commande va charger les donnees, evaluer chaque modele, sauvegarder les metriques dans `results/model_metrics.csv`, et lancer l'application Streamlit sur `http://localhost:8501`.

## Application Streamlit

L'application comprend 5 sections :

1. **Vue d'ensemble** : KPI globaux et evolution temporelle des retards
2. **Exploration des donnees** : saisonnalite, tendances annuelles, top des liaisons
3. **Causes de retard** : repartition des causes et impact COVID
4. **Comparaison des modeles** : tableau comparatif et visualisations
5. **Predicteur interactif** : prediction de retard avec historique de la liaison

## Technologies

Python 3.12, scikit-learn, pandas, numpy, Plotly, Streamlit, joblib

## Auteur

Lenny Sebban - Albert School (B2 Marketing Strategy / Mines Paris PSL)
