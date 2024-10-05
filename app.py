from scipy.stats import chi
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import pickle

with open("./models/rfmodel.pkl", "r+") as file:
    model = pickle.load(file)

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status" : "Todo Gucci"
    }

@app.post("/predict")
async def predict(data : dict):
    
    data = pd.DataFrame([data.dict()])
    prediction = model.predict(data)

    data["Y"] = prediction 
    data.to_csv(path = "./data/new_data.csv", index = False)

    return {
        "prediciton" : int(prediction)
    }


@app.get("/tests")
async def test_dist():

    train_data = pd.read_csv(r"./data/credit_train.csv")