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

# Features proven to generalise — SibSp/Parch subsumed by FamilySize/IsAlone
FEATURES = [
    'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
    'Title', 'FamilySize', 'IsAlone', 'Deck', 'IsChild',
    'AgeBand', 'FareBand',
]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, test, test['PassengerId'].copy()


def print_results(results, n_folds):
    best_name = max(results, key=lambda k: results[k].mean())
    width = 80
    print("\n" + "=" * width)
    print(f"{'Model':<30}  {'Mean':>6}  {'Std':>6}  ({n_folds}-fold scores)")
    print("=" * width)
    for name, scores in results.items():
        flag = "  <-- WINNER" if name == best_name else ""
        print(
            f"{name:<30}  {scores.mean():.4f}  +-{scores.std():.4f}  "
            f"{np.round(scores, 4)}{flag}"
        )
    print("=" * width)
    return best_name


def run():
    print("Loading data...")
    train_raw, test_raw, passenger_ids = load_data()

    print("Engineering features (v3)...")
    train_df, test_df = engineer_features(train_raw.copy(), test_raw.copy())

    # Restrict to the agreed feature set; drop SibSp/Parch
    X      = train_df[FEATURES].copy()
    y      = train_df['Survived'].astype(int)
    X_test = test_df.reindex(columns=FEATURES, fill_value=0)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # --- Model definitions (conservative params to reduce overfitting) ---
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
        n_estimators=200,
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

    voting = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft',
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

    best_name = print_results(results, n_folds=10)
    print(f"\nBest model: {best_name}  ({results[best_name].mean():.4f})")

    # Fit winner on full training data
    best_model = models[best_name]
    best_model.fit(X, y)

    # Classification report (train reference)
    print("\n=== Classification Report (train set) ===")
    print(classification_report(y, best_model.predict(X),
                                target_names=['Not Survived', 'Survived']))

    # Feature importance from RF (cleaner signal than ensemble)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importances — Random Forest v3')
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, 'feature_importance_v3.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved -> {chart_path}")

    # submission_v3.csv
    predictions = best_model.predict(X_test)
    submission  = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    sub_path    = os.path.join(SUBMISSION_DIR, 'submission_v3.csv')
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved -> {sub_path}  ({len(submission)} rows)")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model_v3.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved -> {model_path}")


if __name__ == '__main__':
    run()
