from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

NAME = "YOUR_NAME"
ROLL = "YOUR_ROLLNO"

class InputData(BaseModel):
    Pclass: int
    Sex: int
    Age: float = 30
    Fare: float = 10

@app.get("/health")
def health():
    return {
        "Name": NAME,
        "Roll No": ROLL
    }

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.dict())
    return {
        "prediction": result,
        "Name": NAME,
        "Roll No": ROLL
    }