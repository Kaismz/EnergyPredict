# ⚡ EnergyPredict — Prédiction de la Consommation Énergétique Industrielle

Projet de Machine Learning : prédiction de la consommation énergétique (kWh) d'une aciérie à partir de données de capteurs industriels (puissance réactive, facteur de puissance, type de charge) et de variables temporelles.

## 📊 Dataset

- **Source** : [Steel Industry Energy Consumption Dataset](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption) (Kaggle)
- **Taille** : 35 040 observations, mesures toutes les 15 minutes
- **Période** : Janvier 2018 — Décembre 2018

## 🔧 Pipeline

1. **Exploration** : Analyse des distributions, corrélations, identification du data leakage (CO2 retiré, corrélation 0.99 avec la cible)
2. **Feature Engineering** : Encodage cyclique (sin/cos) des variables temporelles, lags (t-1, t-4, t-96), moyennes glissantes (1h, 24h)
3. **Modélisation** : Benchmark Ridge / Random Forest / XGBoost avec split temporel (80/20)
4. **Optimisation** : GridSearchCV sur XGBoost (72 fits, 3-fold CV)
5. **Interprétabilité** : Analyse SHAP des features les plus influentes

## 📈 Résultats

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 6.625 | 4.401 | 0.9555 |
| Random Forest | 2.358 | 0.954 | 0.9944 |
| XGBoost | 1.971 | 0.965 | 0.9961 |
| **XGBoost optimisé** | **1.750** | **0.927** | **0.9969** |

> RMSE réduit de 74% par rapport à la baseline (Ridge → XGBoost optimisé)

## 🛠 Stack

Python · Pandas · NumPy · Scikit-Learn · XGBoost · SHAP · Matplotlib · Seaborn

## 📂 Structure
```
EnergyPredict/
├── data/                  # Dataset brut et transformé
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                   # Code refactorisé
├── requirements.txt
└── README.md
```

## 🚀 Installation
```bash
git clone https://github.com/Kaismz/EnergyPredict.git
cd EnergyPredict
pip install -r requirements.txt
```
