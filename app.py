from scipy.stats import chi2_contingency, ks_2samp
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
    test_data = pd.read_csv(r"./data/new_data.csv")


    # TODO: implement the test for the old dataset compared to the new dataset
    _,chi2,_,_ = chi2_contingency()

    _,ks = ks_2samp()

    if chi2 < 0.05 or ks < 0.05:
        print("Anda valiendo madre tu modelo")
    else:
        print("No anda valiendo madre tu modelo")

    return {
        "chi2" : float(chi2),
        "ks" : float(ks)
    }