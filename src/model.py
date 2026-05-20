import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE, 'data')
SUBMISSION_DIR = os.path.join(BASE, 'submission')
MODELS_DIR     = os.path.join(BASE, 'models')
PLOTS_DIR      = os.path.join(BASE, 'plots')

for d in (SUBMISSION_DIR, MODELS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

FEATURES = [
    'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
    'Title', 'FamilySize', 'IsAlone', 'Deck', 'IsChild',
    'AgeBand', 'FareBand', 'WomanOrChild',
]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, test, test['PassengerId'].copy()


def print_cv_results(results, n_folds):
    best_name = max(results, key=lambda k: results[k].mean())
    width = 82
    print("\n" + "=" * width)
    print(f"{'Model':<30}  {'Mean':>6}  {'Std':>7}  ({n_folds}-fold scores)")
    print("=" * width)
    for name, scores in results.items():
        flag = "  <-- WINNER" if name == best_name else ""
        print(
            f"{name:<30}  {scores.mean():.4f}  +-{scores.std():.4f}  "
            f"{np.round(scores, 4)}{flag}"
        )
    print("=" * width)
    return best_name


def apply_rules(predictions, test_raw, imputed_age):
    """
    Override borderline predictions with domain-knowledge rules.
    Uses original test_raw for Sex/Pclass, imputed Age for age rule.
    Returns a new predictions array (does not modify in place).
    """
    overridden = predictions.copy()
    sex    = test_raw['Sex'].values
    pclass = test_raw['Pclass'].values
    age    = imputed_age

    # Historical: women in 1st/2nd class had very high survival rates
    mask_survive = (sex == 'female') & (pclass <= 2)
    # Historical: older men in 3rd class had very low survival rates
    mask_die     = (sex == 'male') & (pclass == 3) & (age > 40)

    overridden[mask_survive] = 1
    overridden[mask_die]     = 0

    n_flipped = int((overridden != predictions).sum())
    print(f"  Rule overrides applied: {n_flipped} predictions changed "
          f"({mask_survive.sum()} forced 1, {mask_die.sum()} forced 0)")
    return overridden


def save_submission(passenger_ids, predictions, filename):
    path = os.path.join(SUBMISSION_DIR, filename)
    pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions}).to_csv(path, index=False)
    print(f"  Saved -> {path}  ({len(predictions)} rows)")
    return path


def run():
    print("Loading data...")
    train_raw, test_raw, passenger_ids = load_data()

    print("Engineering features (v4)...")
    train_df, test_df = engineer_features(train_raw.copy(), test_raw.copy())

    X      = train_df[FEATURES].copy()
    y      = train_df['Survived'].astype(int)
    X_test = test_df.reindex(columns=FEATURES, fill_value=0)

    # Imputed ages for rule application (aligned to test_raw row order)
    imputed_age = test_df['Age'].values

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # --- Model definitions ---
    lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

    # RF gets higher weight — it won v3 and has lower variance
    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft',
        weights=[2, 3, 2],
    )

    print("\n=== 10-Fold Cross-Validation ===")
    models = {
        'Logistic Regression': lr,
        'Random Forest':       rf,
        'XGBoost':             xgb,
        'Voting Classifier':   voting,
    }

    results = {}
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        results[name] = scores

    best_name = print_cv_results(results, n_folds=10)

    vc_scores = results['Voting Classifier']
    print(f"\nVoting Classifier (weights=[LR:2, RF:3, XGB:2]):")
    print(f"  10-fold scores : {np.round(vc_scores, 4)}")
    print(f"  Mean accuracy  : {vc_scores.mean():.4f}")
    print(f"  Std            : +-{vc_scores.std():.4f}")

    print(f"\nBest model: {best_name}  ({results[best_name].mean():.4f})")

    # Fit all models on full training data (needed for both submission types)
    print("\nFitting models on full training data...")
    for model in models.values():
        model.fit(X, y)

    # Classification report (train reference)
    best_model = models[best_name]
    print("\n=== Classification Report (train set) ===")
    print(classification_report(y, best_model.predict(X),
                                target_names=['Not Survived', 'Survived']))

    # Feature importance chart from RF
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importances — Random Forest v4')
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, 'feature_importance_v4.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved -> {chart_path}")

    # --- Pure model submission (Voting Classifier) ---
    print("\n=== Generating Submissions ===")
    pred_model = voting.predict(X_test)
    save_submission(passenger_ids, pred_model, 'submission_v4_model.csv')

    # --- Rule-override submission ---
    print("Applying rule-based overrides...")
    pred_rules = apply_rules(pred_model, test_raw, imputed_age)
    save_submission(passenger_ids, pred_rules, 'submission_v4_rules.csv')

    # Save voting model
    model_path = os.path.join(MODELS_DIR, 'best_model_v4.pkl')
    joblib.dump(voting, model_path)
    print(f"Model saved -> {model_path}")


if __name__ == '__main__':
    run()
