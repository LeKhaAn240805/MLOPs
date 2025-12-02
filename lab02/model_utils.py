# model_utils.py
import joblib
import numpy as np

class ModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Loaded model from {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            self.model = None

    def predict(self, features):
        if self.model is None:
            raise ValueError("Model is not loaded.")

        x = np.array(features).reshape(1, -1)

        pred = int(self.model.predict(x)[0])
        prob = float(self.model.predict_proba(x)[0, 1])

        # breast cancer dataset: 0 = malignant, 1 = benign
        label = "malignant" if pred == 0 else "benign"

        return {
            "prediction": pred,
            "probability": prob,
            "label": label
        }
