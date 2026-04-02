# ⚡ EnergyPredict

### Prédiction de la consommation énergétique industrielle par Machine Learning

> Peut-on anticiper la consommation électrique d'une usine à partir de ses données capteurs ?  
> **Spoiler : oui, avec une précision de 99.7%.**

---

## 🎯 Le Problème

L'industrie représente **25% de la consommation énergétique mondiale**. Optimiser cette consommation, c'est réduire les coûts et l'empreinte carbone. Ce projet construit un pipeline ML complet pour prédire la consommation (kWh) d'une aciérie, toutes les 15 minutes, à partir de données capteurs.

## 📊 Le Dataset

| | |
|---|---|
| **Source** | [Steel Industry Energy Consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption) (Kaggle) |
| **Observations** | 35 040 (1 an de mesures, pas de 15 min) |
| **Features** | Puissance réactive, facteur de puissance, type de charge, variables temporelles |
| **Cible** | `Usage_kWh` — consommation énergétique |

## 🔧 Pipeline

```
Données brutes ➜ Exploration ➜ Feature Engineering ➜ Modélisation ➜ Interprétation
```

**1. Exploration** — Analyse des distributions, corrélations. Détection d'un data leakage (CO2 ↔ cible, r=0.99) → variable supprimée.

**2. Feature Engineering** — Encodage cyclique sin/cos (heure, mois), lags temporels (t-1, t-4, t-96), moyennes glissantes (1h, 24h), encodage catégoriel.

**3. Modélisation** — Benchmark de 3 modèles avec split temporel (pas de shuffle, respect de la chronologie).

**4. Optimisation** — GridSearchCV (72 fits, 3-fold CV) sur XGBoost.

**5. Interprétabilité** — Analyse SHAP pour identifier les facteurs clés de consommation.

## 📈 Résultats

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 6.625 | 4.401 | 0.9555 |
| Random Forest | 2.358 | 0.954 | 0.9944 |
| XGBoost | 1.971 | 0.965 | 0.9961 |
| **XGBoost optimisé** | **1.750** | **0.927** | **0.9969** |

> 📉 **RMSE réduit de 74%** entre la baseline Ridge et le modèle final XGBoost optimisé.

### Top 3 des features les plus influentes (SHAP)

1. 🔌 **Puissance réactive** — lien physique direct avec la consommation
2. ⚡ **Facteur de puissance** — indicateur d'efficacité du réseau électrique
3. 🕐 **Lag 15 min (t-1)** — la consommation récente prédit la consommation future

## 🛠 Stack Technique

`Python` · `Pandas` · `NumPy` · `Scikit-Learn` · `XGBoost` · `SHAP` · `Matplotlib` · `Seaborn`

## 📂 Structure du projet

```
EnergyPredict/
├── data/
│   ├── Steel_industry_data.csv      # Dataset brut
│   └── steel_features.csv           # Dataset après feature engineering
├── notebooks/
│   ├── 01_exploration.ipynb          # Analyse exploratoire
│   ├── 02_feature_engineering.ipynb  # Préparation des features
│   └── 03_modeling.ipynb             # Entraînement et évaluation
├── requirements.txt
└── README.md
```

## 🚀 Reproduire le projet

```bash
git clone https://github.com/Kaismz/EnergyPredict.git
cd EnergyPredict
pip install -r requirements.txt
jupyter notebook
```

Exécuter les notebooks dans l'ordre : `01` → `02` → `03`

---


