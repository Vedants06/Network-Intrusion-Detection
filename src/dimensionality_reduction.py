import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def apply_pca(X_train, X_test, n_components=10):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_test_pca

def apply_lda(X_train, y_train, X_test, n_components=4):
    max_components = min(X_train.shape[1], len(np.unique(y_train)) - 1)
    actual_components = min(n_components, max_components)
    lda = LinearDiscriminantAnalysis(n_components=actual_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return lda, X_train_lda, X_test_lda

def apply_svd(X_train, X_test, n_components=10):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)
    return svd, X_train_svd, X_test_svd

def find_optimal_pca_components(X_train, variance_threshold=0.95):
    pca = PCA(random_state=42)
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    optimal_k = int(np.argmax(cumsum >= variance_threshold) + 1)
    return optimal_k, pca.explained_variance_ratio_, cumsum
