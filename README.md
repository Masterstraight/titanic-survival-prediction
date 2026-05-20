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
│   ├── train.csv               # Training data (891 rows, includes 'Survived')
│   └── test.csv                # Test data (418 rows, no 'Survived' column)
│
├── notebooks/
│   └── 01_eda.ipynb            # Full Exploratory Data Analysis notebook
│
├── src/
│   ├── feature_engineering.py  # All feature transformations and imputation
│   └── model.py                # Model training, CV, selection, and submission
│
├── submission/
│   └── submission.csv          # Final predictions ready for Kaggle upload
│
├── models/
│   └── best_model.pkl          # Saved best-performing model (joblib)
│
├── plots/
│   ├── survival_by_sex.png
│   ├── survival_by_pclass.png
│   ├── age_distribution.png
│   ├── survival_by_embarked.png
│   ├── fare_distribution.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
│
├── requirements.txt            # Python dependencies
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
| **Age imputation** | Missing `Age` filled with the median grouped by `Pclass` and `Sex` |
| **Embarked imputation** | Missing `Embarked` filled with the most frequent port (mode) |
| **Fare imputation** | Missing `Fare` filled with the global median fare |
| **Title extraction** | Extracted from `Name` — mapped to: `Mr`, `Mrs`, `Miss`, `Master`, `Rare` |
| **FamilySize** | `SibSp + Parch + 1` — total family members including self |
| **IsAlone** | Binary flag: `1` if `FamilySize == 1`, else `0` |
| **AgeBand** | `Age` binned into 5 equal-width categories (0–4) |
| **FareBand** | `Fare` binned into 4 quantile-based categories (0–3) |
| **Column drops** | `Name`, `Ticket`, `Cabin`, `PassengerId` removed |
| **Label encoding** | `Sex`, `Embarked`, `Title` encoded with `LabelEncoder` |

---

## 🤖 Models Trained

Three classifiers were trained and evaluated using **5-fold Stratified Cross-Validation**:

| Model                | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean Accuracy** | **Std** |
|----------------------|--------|--------|--------|--------|--------|-------------------|---------|
| Logistic Regression  | 0.8101 | 0.8090 | 0.7809 | 0.8034 | 0.8315 | **0.8070**        | ±0.0162 |
| Random Forest        | 0.8101 | 0.7978 | 0.8258 | 0.8315 | 0.8371 | **0.8204** ✅     | ±0.0145 |
| XGBoost              | 0.8268 | 0.8483 | 0.7978 | 0.8034 | 0.8258 | **0.8204**        | ±0.0182 |

> **Best model selected: Random Forest** — tied with XGBoost on mean accuracy but with lower variance (±0.0145 vs ±0.0182).

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/masterstraight009/titanic-survival-prediction.git
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
- Engineer all features from `data/train.csv` and `data/test.csv`
- Train Logistic Regression, Random Forest, and XGBoost with 5-fold CV
- Automatically select the best model
- Save `submission/submission.csv` (418 rows — ready to upload to Kaggle)
- Save `models/best_model.pkl`
- Save `plots/feature_importance.png`

**4. (Optional) Run the EDA notebook**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 🏆 Results

| Submission | Model         | CV Accuracy | Kaggle Public Score |
|------------|---------------|-------------|---------------------|
| v1         | Random Forest | 0.8204      | *TBD after upload*  |

> Upload `submission/submission.csv` to the [Titanic competition](https://www.kaggle.com/competitions/titanic/submit) to get your public leaderboard score.

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
