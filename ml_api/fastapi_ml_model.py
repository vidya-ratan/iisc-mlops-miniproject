from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load the pre-trained linear regression model
# model_path = "pretrained_model.pkl"
# model = joblib.load(model_path)
import os

# Print the current working directory
# print("Current working directory:", os.getcwd())
dataset = pd.read_csv('ml_api/Salary_Data.csv')
X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
y = dataset.iloc[:, 1].values #get array of dataset in column 1st

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# class PredictionInput(BaseModel):
#     feature_values: list

@app.get("/predict")
def predict():
    try:
        # # Convert input data to NumPy array
        # features = np.array(input_data.feature_values).reshape(1, -1)

        # # Make prediction using the pre-trained model
        # prediction = model.predict(features)[0]

        y_pred = regressor.predict(X_test).tolist()

        return {"prediction": y_pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application using Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
