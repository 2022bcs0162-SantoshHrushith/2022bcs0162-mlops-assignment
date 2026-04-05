import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(data):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return int(pred)