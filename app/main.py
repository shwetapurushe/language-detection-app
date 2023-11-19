from fastapi import FastAPI
from pydantic import BaseModel
from app.model.predict import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    return {"API health check": "OK"}

@app.get("/about")
def about():
    return {"Data" : "Returning about page"}

@app.post("/language", response_model=PredictionOut)
def language(payload: TextIn):
    predicted_language = predict_pipeline(payload.text)
    return {"language": predicted_language.upper()}