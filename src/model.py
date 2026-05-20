import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Make src importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from feature_engineering import engineer_features

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
SUBMISSION_DIR = os.path.join(BASE, 'submission')
MODELS_DIR = os.path.join(BASE, 'models')
PLOTS_DIR = os.path.join(BASE, 'plots')

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    passenger_ids = test['PassengerId'].copy()
    return train, test, passenger_ids


def run():
    print("Loading data...")
    train_raw, test_raw, passenger_ids = load_data()

    print("Engineering features...")
    train_df, test_df = engineer_features(train_raw.copy(), test_raw.copy())

    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived'].astype(int)
    X_test = test_df.copy()

    # Align columns
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, use_label_encoder=False,
                                  eval_metric='logloss', random_state=42, verbosity=0),
    }

    results = {}
    print("\n=== 5-Fold Cross-Validation Results ===")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = mean_acc
        print(f"{name:25s}  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}  Folds: {np.round(scores, 4)}")

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]:.4f})")

    # Fit best model on full training data
    best_model.fit(X, y)

    # Classification report on training set (for reference)
    y_pred_train = best_model.predict(X)
    print("\n=== Classification Report (train set) ===")
    print(classification_report(y, y_pred_train, target_names=['Not Survived', 'Survived']))

    # Feature importance plot
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances.sort_values().plot(kind='barh', figsize=(8, 6), title=f'Feature Importances ({best_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
        print(f"Feature importance chart saved to plots/feature_importance.png")

    # Submission
    predictions = best_model.predict(X_test)
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}  ({len(submission)} rows)")

    # Save best model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")


if __name__ == '__main__':
    run()
