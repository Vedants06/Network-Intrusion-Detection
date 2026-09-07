import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_svm_classifier(X_train, y_train, kernel="rbf", C=1.0, gamma="scale", degree=3, probability=False):
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=probability,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train, kernel="rbf", C=1.0, epsilon=0.1):
    model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon
    )
    model.fit(X_train, y_train)
    return model

def train_multiclass_svm(X_train, y_train, decision_function_shape="ovr", kernel="rbf", C=1.0):
    model = SVC(
        kernel=kernel,
        C=C,
        decision_function_shape=decision_function_shape,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def get_support_vectors_info(model):
    return {
        "n_support": model.n_support_.tolist(),
        "total_support_vectors": int(np.sum(model.n_support_))
    }

def evaluate_svr_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
