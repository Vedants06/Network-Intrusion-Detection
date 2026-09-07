# src/feature_engineering.py
"""
Feature engineering utilities: correlation analysis, feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def get_correlation_matrix(X, feature_names, threshold=0.95):
    """
    Compute correlation matrix and find highly correlated pairs.

    Returns:
        corr_matrix: DataFrame correlation matrix
        high_corr_pairs: list of (feat1, feat2, correlation) tuples
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    round(corr_val, 4)
                ))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    return corr_matrix, high_corr_pairs


def select_top_k_features(X, y, k=20, method='f_classif'):
    """
    Select top K features using statistical tests.

    Args:
        X: Feature matrix
        y: Target labels
        k: Number of features to select
        method: 'f_classif' or 'mutual_info'

    Returns:
        X_selected: Reduced feature matrix
        selected_indices: Indices of selected features
        scores: Feature scores
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_

    return X_selected, selected_indices, scores


def get_feature_importance_df(feature_names, scores, top_n=20):
    """Create a sorted DataFrame of feature importances."""
    df = pd.DataFrame({
        'feature': feature_names,
        'score': scores
    })
    df = df.sort_values('score', ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    return df
