from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import pickle

app = FastAPI()

@app.get("/health")
def health():
    return {
        "status" : "Todo Gucci"
    }


