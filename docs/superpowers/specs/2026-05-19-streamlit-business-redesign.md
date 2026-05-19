# Streamlit Business Redesign — Spec
**Date:** 2026-05-19
**Projet:** SNCF TGV Delay Predictor — ml-poc-project
**Audience cible:** Comité de direction (non-technique)
**Objectif:** Transformer l'app ML en outil de présentation business

---

## Contexte

L'app Streamlit actuelle (`src/app.py`) est orientée ML : elle expose la comparaison des modèles, les métriques techniques, et la structure de features. Pour une présentation au comité de direction, on repositionne l'angle sur **l'outil et ses insights**, pas sur la technique.

---

## Structure des pages

### 1. Contexte & Enjeux
- **KPIs globaux** : retard moyen (min), nombre de liaisons analysées, nombre d'observations, pire liaison
- **Courbe d'évolution temporelle** : retard moyen par mois/année (thème clair)
- **Encart modèle** (discret, en bas) : "Modèle utilisé : Gradient Boosting — Précision R² = 0.68 — Erreur moyenne : X min"
- Remplace les anciennes pages "Vue d'ensemble" + "Exploration des données"

### 2. Liaisons à risque
- **Top 15 liaisons** les plus retardées : bar chart horizontal, trié par retard moyen décroissant
- **Table interactive** filtrables par gare de départ (st.selectbox)
- Donne aux décideurs une liste concrète et actionnables des routes prioritaires

### 3. Comprendre les retards
- **Répartition des causes** : graphique camembert ou bar chart horizontal (causes existantes)
- **Saisonnalité** : retard moyen par mois (bar chart)
- Renommé "Comprendre les retards" (exit "Causes de retard")
- Supprime les sections trop techniques (avant/après COVID conservé si pertinent)

### 4. Prédire un retard
- **Conservé tel quel** — page prédicteur déjà réécrite avec date picker, heure, routes valides
- Léger ajustement : libellés plus business ("Estimation du retard" plutôt que "Retard prédit")

### 5. Synthèse
- **3 blocs insight** : faits marquants tirés des données (ex: "Juillet = mois le plus à risque", "Paris-Lyon : liaison la plus impactée", "Précision de prédiction : X min d'erreur moyenne")
- **Section "Et ensuite ?"** : 3 pistes concrètes (déploiement en production, enrichissement données temps réel, système d'alertes automatiques)
- Style "slide de conclusion" — page qu'on laisse à l'écran lors des questions

---

## Design

### Palette
| Élément | Couleur |
|---|---|
| Fond principal | `#ffffff` |
| Sidebar fond | `#f8f9fa` |
| Titres | `#1a3a5c` (bleu marine) |
| Accents / bordures | `#2563eb` (bleu) |
| Texte secondaire | `#6b7280` (gris) |
| Succès / positif | `#059669` (vert) |

### Composants
- **Cartes métriques** : fond blanc, bordure gauche 4px bleue, ombre légère (`box-shadow`)
- **Graphiques** : tous en thème `plotly` clair (exit `plotly_dark`)
- **Sidebar** : fond gris très clair, emoji 🚄 + titre projet, navigation radio sans label
- **Titres de section** : bleu marine, underline bleu, taille 1.4rem

### CSS custom
Réécriture du bloc `<style>` actuel pour refléter le thème clair business.

---

## Ce qu'on supprime

| Ancien | Action |
|---|---|
| Page "Exploration des données" | Supprimée — contenu fusionné dans "Contexte & Enjeux" |
| Page "Comparaison des modèles" | Supprimée — remplacée par encart discret dans "Contexte & Enjeux" |
| Thème `plotly_dark` | Remplacé par thème clair partout |
| Libellés techniques ("R2", "RMSE", "MAE") | Remplacés par langage business |

---

## Ce qu'on garde

- Page prédicteur (Prédire un retard) — déjà bien refaite
- Données existantes — aucun changement au backend
- Structure `build_app()` dans `src/app.py`

---

## Fichiers impactés

- `src/app.py` — réécriture complète de la navigation et du contenu des pages
- Aucun autre fichier backend modifié

---

## Critères de succès

- Un non-technicien peut naviguer dans l'app sans explication
- La page "Synthèse" peut être laissée à l'écran pendant les questions
- Aucune mention de "Random Forest", "KNN", "RMSE", "feature vector" dans les pages principales
- Le prédicteur fonctionne et retourne une estimation compréhensible
