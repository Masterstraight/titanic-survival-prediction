import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    rare_titles = {'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'}
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    }
    if title in rare_titles:
        return 'Rare'
    return title_map.get(title, 'Rare')


def engineer_features(train_df, test_df):
    combined = pd.concat([train_df.drop(columns=['Survived']), test_df], keys=['train', 'test'])

    # Fill missing Age with median grouped by Pclass and Sex
    age_medians = combined.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    combined['Age'] = combined['Age'].fillna(age_medians)

    # Fill missing Embarked with mode
    combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])

    # Fill missing Fare with median
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())

    # Title from Name
    combined['Title'] = combined['Name'].apply(extract_title)

    # Family features
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)

    # Age and Fare bands
    combined['AgeBand'] = pd.cut(combined['Age'], bins=5, labels=False)
    combined['FareBand'] = pd.qcut(combined['Fare'], q=4, labels=False, duplicates='drop')

    # Drop unused columns
    combined.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

    # Encode categoricals
    cat_cols = ['Sex', 'Embarked', 'Title']
    le = LabelEncoder()
    for col in cat_cols:
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_clean = combined.loc['train'].copy()
    test_clean = combined.loc['test'].copy()

    train_clean['Survived'] = train_df['Survived'].values

    return train_clean, test_clean
