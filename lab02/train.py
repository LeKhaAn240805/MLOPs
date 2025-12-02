# train.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def main():
    # 1. Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Pipeline: scaling -> feature selection (k best) -> classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=10)),  # chọn 10 đặc trưng tốt nhất
        ("clf", RandomForestClassifier(random_state=42))
    ])

    # 4. Optional: grid search to tune RF hyperparams
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 5, 10],
        # "select__k": [8, 10, 12]  # nếu muốn tune số đặc trưng
    }
    gs = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)

    # 5. Evaluate on test set
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # 6. Save the pipeline
    joblib.dump(best_model, "model_pipeline.joblib")
    print("Saved model to model_pipeline.joblib")

if __name__ == "__main__":
    main()