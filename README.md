# SNCF TGV Delay Predictor

Proof of Concept de Machine Learning pour predire le retard moyen a l'arrivee des TGV en France, a partir des donnees Open Data SNCF.

## Objectif

Predire le **retard moyen a l'arrivee** (en minutes) d'une liaison TGV donnee, pour un mois donne, en exploitant les donnees historiques de regularite publiees par la SNCF.

## Dataset

- **Source** : [SNCF Open Data - Regularite mensuelle TGV par liaisons](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/)
- **Volume** : ~12 000 observations (2018-2026)
- **Granularite** : mensuelle, par liaison gare a gare
- **Variable cible** : retard moyen de tous les trains a l'arrivee (minutes)

Le fichier CSV (`sncf_retards.csv`) n'est pas versionne. Pour reproduire le projet, telechargez-le depuis le lien ci-dessus et placez-le a la racine du depot.

## Resultats

### V1 — Baseline (7 features)

| Modele | MAE (min) | RMSE (min) | R2 |
|--------|-----------|------------|------|
| Gradient Boosting | 1.89 | 2.74 | 0.50 |
| Random Forest | 1.89 | 2.77 | 0.49 |
| K-Nearest Neighbors | 2.17 | 3.06 | 0.38 |
| Linear Regression | 2.66 | 3.59 | 0.15 |

### V2 — Feature engineering enrichi (17 features)

| Modele | MAE (min) | RMSE (min) | R2 |
|--------|-----------|------------|------|
| Gradient Boosting | ~1.5 | ~2.2 | **0.68** |
| Random Forest | — | — | ~0.65 |
| K-Nearest Neighbors | — | — | ~0.50 |
| Linear Regression | — | — | ~0.18 |

Le **Gradient Boosting** est le modele retenu. Le passage V1 -> V2 ameliore le R2 de +0.18 points grace aux features de saisonnalite cyclique, taux d'annulation et repartition des causes de retard.

## Notebooks

| Notebook | Role |
|----------|------|
| `01_eda.ipynb` | Analyse exploratoire (distribution de la cible, saisonnalite, top liaisons, correlations) |
| `02_preprocessing_training.ipynb` | Baseline V1 : preprocessing, encodage des gares, entrainement et evaluation de 4 modeles |
| `03_feature_engineering.ipynb` | Feature engineering V2 : 17 features, GridSearchCV, cross-validation 5 folds, importance des features |

Executez les notebooks dans cet ordre pour reproduire les resultats.

## Structure du projet

    ml-poc-project/
    ├── data/                  # Repertoire cible pour les donnees (vide, non versionne)
    ├── models/                # Modeles entraines (.joblib) et metriques CSV
    ├── notebooks/
    │   ├── 01_eda.ipynb                      # Analyse exploratoire
    │   ├── 02_preprocessing_training.ipynb   # Baseline V1
    │   └── 03_feature_engineering.ipynb      # Feature engineering V2
    ├── plots/                 # Graphiques exportes
    ├── results/               # model_metrics.csv
    ├── scripts/
    │   └── main.py            # Pipeline complet (entrainement + sauvegarde metriques)
    ├── src/
    │   ├── config.py          # Chemins, constantes, registre des modeles
    │   ├── data.py            # Chargement, nettoyage et split train/test (V2)
    │   ├── metrics.py         # Calcul et sauvegarde des metriques
    │   ├── model_io.py        # Chargement/sauvegarde des modeles joblib
    │   ├── results.py         # Agregation et export des resultats
    │   └── app.py             # Application Streamlit (5 sections)
    ├── tests/
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Probleme de data leakage identifie

La colonne `col_retard_dep` ("Retard moyen de tous les trains au depart") et la feature derivee `pct_retard_depart` sont **contemporaines de la variable cible** : elles decrivent le meme lot de trajets, le meme mois. Les inclure dans un modele de prediction prospective constitue du label leakage.

Ces deux features sont **exclues par defaut** dans `src/data.py` (flag `INCLUDE_DEPARTURE_DELAY_FEATURES = False`). Le R2 de 0.68 est mesure sans elles. Elles peuvent etre reactivees uniquement dans un contexte d'explication contemporaine (ex. : comprendre pourquoi un mois est en retard apres coup), a condition de le documenter explicitement dans la fiche modele.

## Installation et execution

### 1. Cloner le depot

    git clone git@github.com:lenny1890/ml-poc-project.git
    cd ml-poc-project

### 2. Creer l'environnement virtuel

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 3. Telecharger les donnees

Telechargez le CSV depuis [SNCF Open Data](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/) et placez-le a la racine du depot : `sncf_retards.csv`.

### 4. Reproduire les modeles

Executez les notebooks dans l'ordre (`01_eda`, `02_preprocessing_training`, `03_feature_engineering`) ou lancez le pipeline complet :

    python scripts/main.py

Cette commande charge les donnees, evalue chaque modele et sauvegarde les metriques dans `results/model_metrics.csv`.

## Lancer l'application

    streamlit run src/app.py

L'application s'ouvre sur `http://localhost:8501` et comprend 5 sections :

1. **Vue d'ensemble** : KPI globaux et evolution temporelle des retards
2. **Exploration des donnees** : saisonnalite, tendances annuelles, top des liaisons
3. **Causes de retard** : repartition des 6 causes et impact COVID
4. **Comparaison des modeles** : tableau comparatif et visualisations (lit `results/model_metrics.csv`)
5. **Predicteur interactif** : selection d'une liaison et d'un mois, prediction et historique de la liaison

> **Prerequis** : les modeles `.joblib` doivent etre generes avant de lancer l'application (etape 4).

## Technologies

Python 3.12, scikit-learn, pandas, numpy, Plotly, Streamlit, joblib

## Auteur

Lenny Sebban - Albert School (B2 Marketing Strategy / Mines Paris PSL)
