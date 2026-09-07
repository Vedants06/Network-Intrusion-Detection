from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def train_decision_tree_classifier(X_train, y_train, criterion="gini", max_depth=None):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_regressor(X_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model
