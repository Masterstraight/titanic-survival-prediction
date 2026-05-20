import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Titles that indicate female social status (high survival)
_RARE_FEMALE = {'Lady', 'Countess', 'Dona'}
# Titles that indicate male professional/military status
_RARE_MALE = {'Rev', 'Col', 'Major', 'Capt', 'Don', 'Sir', 'Jonkheer'}

_TITLE_MAP = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Dr': 'Dr',
}


def _extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    if title in _RARE_FEMALE:
        return 'Rare_F'
    if title in _RARE_MALE:
        return 'Rare_M'
    return _TITLE_MAP.get(title, 'Rare_M')


def _extract_deck(cabin):
    if pd.isna(cabin) or str(cabin).strip() == '':
        return 'U'
    letter = str(cabin).strip()[0].upper()
    # Valid Titanic decks: A–G, T. Anything else → U
    return letter if letter in set('ABCDEFGT') else 'U'


def engineer_features(train_df, test_df):
    combined = pd.concat(
        [train_df.drop(columns=['Survived']), test_df],
        keys=['train', 'test'],
    )

    # --- Imputation ---
    age_medians = combined.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    combined['Age'] = combined['Age'].fillna(age_medians)
    combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
    combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())

    # --- Title (refined) ---
    combined['Title'] = combined['Name'].apply(_extract_title)

    # --- Deck from Cabin (before dropping Cabin) ---
    combined['Deck'] = combined['Cabin'].apply(_extract_deck)

    # --- Ticket frequency: passengers sharing the same ticket ---
    combined['TicketFrequency'] = combined.groupby('Ticket')['Ticket'].transform('count')

    # --- Family features ---
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)

    # --- Derived fare / age features ---
    combined['FarePerPerson'] = combined['Fare'] / combined['FamilySize']
    combined['IsChild'] = (combined['Age'] < 16).astype(int)

    # --- Binned features ---
    combined['AgeBand'] = pd.cut(combined['Age'], bins=5, labels=False)
    combined['FareBand'] = pd.qcut(combined['Fare'], q=4, labels=False, duplicates='drop')

    # --- Drop raw columns no longer needed ---
    combined.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

    # --- Encode categoricals ---
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'Title', 'Deck']:
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_clean = combined.loc['train'].copy()
    test_clean = combined.loc['test'].copy()
    train_clean['Survived'] = train_df['Survived'].values

    return train_clean, test_clean
