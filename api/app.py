from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict as get_prediction

app = FastAPI(title="Twitter Sentiment Analysis API")

# Request schema
class TweetInput(BaseModel):
    tweet: str

# Health check 
@app.get("/")
def health():
    return {"message": "Welcome to Twitter Sentiment Analysis"}

# Predict endpoint 
@app.post("/predict/")
def predict_tweet(data: TweetInput):
    result = get_prediction(data.tweet)
    return result