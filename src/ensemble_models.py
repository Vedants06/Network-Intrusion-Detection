import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_decision_stump(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0):
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_bagging(X_train, y_train, n_estimators=50, max_samples=0.8):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_subagging(X_train, y_train, n_estimators=50, max_samples=0.5):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=15):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_voting_classifier(estimators, X_train, y_train, voting="hard"):
    model = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_stacking_classifier(estimators, final_estimator, X_train, y_train, cv=5):
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_kfold_cv(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "scores": scores.tolist()
    }
