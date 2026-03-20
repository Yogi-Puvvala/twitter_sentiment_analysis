from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict as get_prediction

app = FastAPI(title="Twitter Sentiment Analysis API")

class TweetInput(BaseModel):
    tweet: str

@app.get("/")
def health():
    return {"message": "Welcome to Twitter Sentiment Analysis"}

@app.post("/predict/")
def predict_tweet(data: TweetInput):
    result = get_prediction(data.tweet)
    return result