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


