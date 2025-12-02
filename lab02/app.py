# app.py

from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------------------
# MÔ HÌNH GIẢ LẬP (Dummy Model)
# Ở lab thật sẽ thay bằng model sklearn hoặc joblib
# -----------------------------------------
class DummyModel:
    def predict(self, x):
        # Quy ước: nếu tổng giá trị > 50 → lớp 1, ngược lại lớp 0
        return [1 if sum(x[0]) > 50 else 0]

    def predict_proba(self, x):
        # giả lập xác suất
        prob = min(1.0, sum(x[0]) / 100)
        return [[1 - prob, prob]]

model = DummyModel()

# -----------------------------------------
# KHAI BÁO FASTAPI
# -----------------------------------------
app = FastAPI(
    title="FastAPI ML Demo",
    description="Demo triển khai mô hình ML bằng FastAPI (dummy model)",
    version="1.0"
)

# -----------------------------------------
# ĐỊNH NGHĨA INPUT DỮ LIỆU CHO /predict
# -----------------------------------------
class PredictRequest(BaseModel):
    features: list[float]     # Danh sách các số thực (list float)

class PredictResponse(BaseModel):
    prediction: int
    probability: float


# -----------------------------------------
# ENDPOINT ROOT "/"
# -----------------------------------------
@app.get("/")
def index():
    info = {
        "model": "DummyModel — mô hình giả lập để demo FastAPI",
        "input_format": "{ 'features': [list các số float] }",
        "predict_endpoint": "/predict",
        "description": "Dùng /docs để test API"
    }
    return info


# -----------------------------------------
# ENDPOINT PREDICT "/predict"
# -----------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    x = [req.features]             # phải bọc lại thành mảng 2D cho giống sklearn
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0][1]

    return PredictResponse(
        prediction=pred,
        probability=float(prob)
    )
