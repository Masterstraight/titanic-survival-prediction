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
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE, 'data')
SUBMISSION_DIR  = os.path.join(BASE, 'submission')
MODELS_DIR      = os.path.join(BASE, 'models')
PLOTS_DIR       = os.path.join(BASE, 'plots')

for d in (SUBMISSION_DIR, MODELS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, test, test['PassengerId'].copy()


def _fold_scores_from_search(search, n_splits=5):
    """Extract per-fold scores for the best param set from a fitted RandomizedSearchCV."""
    idx = search.best_index_
    return np.array([
        search.cv_results_[f'split{i}_test_score'][idx]
        for i in range(n_splits)
    ])


def tune_random_forest(X, y, cv):
    print("  Tuning Random Forest with RandomizedSearchCV (40 iters × 5 folds)...")
    param_dist = {
        'n_estimators':      [100, 200, 300, 500],
        'max_depth':         [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      ['sqrt', 'log2'],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist,
        n_iter=40, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    scores = _fold_scores_from_search(search)
    print(f"  Best RF params: {search.best_params_}")
    return search.best_estimator_, scores


def tune_xgboost(X, y, cv):
    print("  Tuning XGBoost with RandomizedSearchCV (40 iters × 5 folds)...")
    param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [3, 4, 5, 6, 7],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma':            [0, 0.1, 0.2],
    }
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0),
        param_dist,
        n_iter=40, cv=cv, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    scores = _fold_scores_from_search(search)
    print(f"  Best XGB params: {search.best_params_}")
    return search.best_estimator_, scores


def run():
    print("Loading data...")
    train_raw, test_raw, passenger_ids = load_data()

    print("Engineering features (v2)...")
    train_df, test_df = engineer_features(train_raw.copy(), test_raw.copy())

    X      = train_df.drop(columns=['Survived'])
    y      = train_df['Survived'].astype(int)
    X_test = test_df.reindex(columns=X.columns, fill_value=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== Training & Tuning Models ===")

    # 1. Logistic Regression (baseline)
    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    lr.fit(X, y)

    # 2. Random Forest — hyperparameter search
    best_rf, rf_scores = tune_random_forest(X, y, cv)

    # 3. XGBoost — hyperparameter search
    best_xgb, xgb_scores = tune_xgboost(X, y, cv)

    # 4. Soft Voting Ensemble
    print("  Training Voting Classifier (LR + tuned RF + tuned XGB, soft voting)...")
    voting = VotingClassifier(
        estimators=[
            ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
            ('rf',  best_rf),
            ('xgb', best_xgb),
        ],
        voting='soft',
    )
    voting_scores = cross_val_score(voting, X, y, cv=cv, scoring='accuracy')

    # --- Results table ---
    results = {
        'Logistic Regression':  lr_scores,
        'Random Forest (tuned)': rf_scores,
        'XGBoost (tuned)':       xgb_scores,
        'Voting Classifier':     voting_scores,
    }
    best_name  = max(results, key=lambda k: results[k].mean())
    best_mean  = results[best_name].mean()

    print("\n" + "=" * 75)
    print(f"{'Model':<30}  {'Mean':>6}  {'Std':>6}  Folds")
    print("=" * 75)
    for name, scores in results.items():
        flag = " <-- WINNER" if name == best_name else ""
        print(
            f"{name:<30}  {scores.mean():.4f}  ±{scores.std():.4f}  "
            f"{np.round(scores, 4)}{flag}"
        )
    print("=" * 75)
    print(f"\nBest model: {best_name}  ({best_mean:.4f})")

    # Fit winner on full training data
    model_map = {
        'Logistic Regression':   lr,
        'Random Forest (tuned)': best_rf,
        'XGBoost (tuned)':       best_xgb,
        'Voting Classifier':     voting,
    }
    best_model = model_map[best_name]
    best_model.fit(X, y)

    # Classification report (train set — for reference only)
    print("\n=== Classification Report (train set) ===")
    print(classification_report(y, best_model.predict(X),
                                target_names=['Not Survived', 'Survived']))

    # Feature importance chart (from tuned RF regardless of winner)
    importances = pd.Series(best_rf.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importances — Tuned Random Forest (v2)')
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, 'feature_importance_v2.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to {chart_path}")

    # submission_v2.csv
    predictions   = best_model.predict(X_test)
    submission    = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    sub_path      = os.path.join(SUBMISSION_DIR, 'submission_v2.csv')
    submission.to_csv(sub_path, index=False)
    print(f"\nSubmission saved -> {sub_path}  ({len(submission)} rows)")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model_v2.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved -> {model_path}")


if __name__ == '__main__':
    run()
