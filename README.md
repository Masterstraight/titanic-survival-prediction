# 🚢 Titanic Survival Prediction
### *Can you predict who survives the Titanic disaster using machine learning?*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic%20Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)

---

## 📖 About the Challenge

The [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) is the classic entry-point for machine learning. Using passenger data (age, class, gender, family size, etc.) from the 1912 Titanic disaster, the goal is to **predict whether a given passenger survived (1) or did not survive (0)**.

- **Training set:** 891 passengers with known outcomes  
- **Test set:** 418 passengers — submit your predictions to Kaggle  
- **Evaluation metric:** Classification Accuracy  

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   ├── train.csv                     # Training data (891 rows, includes 'Survived')
│   └── test.csv                      # Test data (418 rows, no 'Survived' column)
│
├── notebooks/
│   └── 01_eda.ipynb                  # Full Exploratory Data Analysis notebook
│
├── src/
│   ├── feature_engineering.py        # All feature transformations and imputation
│   └── model.py                      # Model training, CV, ensemble, and submission
│
├── submission/
│   ├── submission.csv                # v1 — baseline Random Forest
│   ├── submission_v2.csv             # v2 — tuned XGBoost (hyperparameter search)
│   ├── submission_v3.csv             # v3 — conservative RF, 10-fold CV
│   ├── submission_v4_model.csv       # v4 — weighted Voting Classifier (pure model)
│   └── submission_v4_rules.csv       # v4 — model + rule-based overrides
│
├── models/
│   ├── best_model.pkl                # v1 model
│   ├── best_model_v2.pkl             # v2 model
│   ├── best_model_v3.pkl             # v3 model
│   └── best_model_v4.pkl             # v4 model (current best)
│
├── plots/
│   ├── survival_by_sex.png
│   ├── survival_by_pclass.png
│   ├── age_distribution.png
│   ├── survival_by_embarked.png
│   ├── fare_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png        # v1
│   ├── feature_importance_v2.png
│   ├── feature_importance_v3.png
│   └── feature_importance_v4.png     # current
│
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset is provided by Kaggle. Below is the full data dictionary:

| Feature       | Type        | Description |
|---------------|-------------|-------------|
| `PassengerId` | Integer     | Unique identifier for each passenger |
| `Survived`    | Integer     | Target variable — 0 = Did not survive, 1 = Survived |
| `Pclass`      | Integer     | Ticket class — 1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower) |
| `Name`        | String      | Full name of the passenger (includes title, e.g., Mr., Mrs.) |
| `Sex`         | String      | Passenger gender — `male` or `female` |
| `Age`         | Float       | Age in years; fractional if < 1. Estimated ages are in the form xx.5 |
| `SibSp`       | Integer     | Number of siblings / spouses aboard the Titanic |
| `Parch`       | Integer     | Number of parents / children aboard the Titanic |
| `Ticket`      | String      | Ticket number |
| `Fare`        | Float       | Passenger fare (in British pounds) |
| `Cabin`       | String      | Cabin number (many missing values) |
| `Embarked`    | String      | Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton |

---

## ⚙️ Feature Engineering

All transformations are implemented in `src/feature_engineering.py`:

| Transformation | Description |
|----------------|-------------|
| **Age imputation** | Missing `Age` filled with median grouped by `Pclass` and `Sex` |
| **Embarked imputation** | Missing `Embarked` filled with mode |
| **Fare imputation** | Missing `Fare` filled with global median |
| **Title extraction** | Parsed from `Name` — groups: `Mr`, `Mrs`, `Miss`, `Master`, `Dr`, `Rare_F`, `Rare_M` |
| **Deck** | First letter of `Cabin` (A–G, T); `U` for unknown (replaces dropping Cabin) |
| **FamilySize** | `SibSp + Parch + 1` — total aboard including self |
| **IsAlone** | `1` if `FamilySize == 1`, else `0` |
| **IsChild** | `1` if `Age < 16`, else `0` |
| **WomanOrChild** | `1` if `Sex == female` OR `Age < 16` — encodes "women and children first" rule |
| **AgeBand** | `Age` binned into 5 equal-width categories |
| **FareBand** | `Fare` binned into 4 quantile-based categories |
| **Column drops** | `Name`, `Ticket`, `Cabin`, `PassengerId` removed |
| **Label encoding** | `Sex`, `Embarked`, `Title`, `Deck` encoded with `LabelEncoder` |

> Features removed after v2 overfitting analysis: `TicketFrequency`, `FarePerPerson` (added noise, hurt generalisation).

---

## 🤖 Models & Training Strategy

Training uses **10-fold Stratified Cross-Validation** with conservative hyperparameters to minimise the train/CV gap.

### Current model parameters (v4)

| Model | Key Parameters |
|---|---|
| Logistic Regression | `C=0.1`, `max_iter=1000` |
| Random Forest | `n_estimators=500`, `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=4` |
| XGBoost | `n_estimators=500`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `gamma=1`, `reg_alpha=0.1` |
| **Voting Classifier** | Soft voting, weights `[LR:2, RF:3, XGB:2]` |

### v4 — 10-Fold CV Results

| Model | Mean Accuracy | Std |
|---|---|---|
| Logistic Regression | 0.8159 | ±0.0364 |
| Random Forest | 0.8294 | ±0.0209 |
| XGBoost | 0.8316 | ±0.0299 |
| **Voting Classifier** | **0.8327** ✅ | ±0.0314 |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Masterstraight/titanic-survival-prediction.git
cd titanic-survival-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full ML pipeline**
```bash
python src/model.py
```

This will:
- Engineer all features (including `WomanOrChild`, `Deck`, `IsChild`) from `data/train.csv` and `data/test.csv`
- Evaluate Logistic Regression, Random Forest, XGBoost, and Voting Classifier with 10-fold CV
- Fit the winning Voting Classifier on full training data
- Save `submission/submission_v4_model.csv` — pure model predictions (418 rows)
- Save `submission/submission_v4_rules.csv` — model + rule-based overrides
- Save `models/best_model_v4.pkl`
- Save `plots/feature_importance_v4.png`

**4. (Optional) Run the EDA notebook**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 🏆 Kaggle Results

| Version | Model | CV Accuracy | Kaggle Public Score | Notes |
|---------|-------|-------------|---------------------|-------|
| v1 | Random Forest (baseline) | 0.8204 | 0.74641 | 5-fold CV, basic features |
| v2 | XGBoost (RandomizedSearchCV) | 0.8418 | 0.76555 | Overfit — large train/CV gap |
| v3 | Random Forest (conservative) | 0.8327 | 0.77990 | 10-fold CV, removed noisy features |
| v4 | Voting Classifier (weighted) | 0.8327 | *pending* | WomanOrChild feature, RF weight=3 |

> **Overfitting fix:** Train accuracy dropped from 0.90 (v2) to 0.86 (v3/v4), halving the train/CV gap and improving generalisation.

---

## 🛠️ Tech Stack

| Library        | Purpose |
|----------------|---------|
| `Python 3.8+`  | Core language |
| `pandas`       | Data manipulation and analysis |
| `numpy`        | Numerical operations |
| `scikit-learn` | Machine learning models, preprocessing, cross-validation |
| `XGBoost`      | Gradient boosted tree classifier |
| `matplotlib`   | Plotting and chart generation |
| `seaborn`      | Statistical data visualization |
| `joblib`       | Model serialization (save/load `.pkl`) |
| `Jupyter`      | Interactive notebooks for EDA |

---

## 👤 Author

**Built by Abdulrahim**  
Follow the journey on Twitter: [@Gigly_NG](https://twitter.com/Gigly_NG)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
