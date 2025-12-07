# ============================================================
#   BÀI LÀM HOÀN CHỈNH – PHÂN LOẠI THU NHẬP ADULT INCOME
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. TẢI DỮ LIỆU
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]

df = pd.read_csv(url, names=columns, na_values=' ?')
print("Dữ liệu ban đầu:")
print(df.head())


# ============================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ============================================================

# Xóa giá trị thiếu
df = df.dropna()

# Tách features và label
X = df.drop('income', axis=1)
y = df['income']

# Nhãn hóa y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ============================================================
# 3. TRƯỜNG HỢP 1 – KHÔNG TIỀN XỬ LÝ
# ============================================================

print("\n=== TRƯỜNG HỢP 1: KHÔNG TIỀN XỬ LÝ ===")

results_simple = {}
models_simple = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Chia train/test
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

for name, model in models_simple.items():
    try:
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        results_simple[name] = {
            "Accuracy": accuracy_score(y_test_s, y_pred),
            "Precision": precision_score(y_test_s, y_pred),
            "Recall": recall_score(y_test_s, y_pred),
            "F1": f1_score(y_test_s, y_pred)
        }
    except Exception as e:
        results_simple[name] = f"Error: {e}"

print(results_simple)


# ============================================================
# 4. TRƯỜNG HỢP 2 – TIỀN XỬ LÝ ĐẦY ĐỦ
# ============================================================

print("\n=== TRƯỜNG HỢP 2: CÓ TIỀN XỬ LÝ ===")

numeric_features = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
categorical_features = [c for c in X.columns if c not in numeric_features]

# Tiền xử lý
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Các mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results_processed = {}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Huấn luyện
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results_processed[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

print(results_processed)


# ============================================================
# 6. IN RA CONFUSION MATRIX MẪU CHO MODEL TỐT NHẤT
# ============================================================

best_model_name = max(results_processed, key=lambda m: results_processed[m]["F1"])
print(f"\nModel tốt nhất: {best_model_name}")

best_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', models[best_model_name])
])

best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))
