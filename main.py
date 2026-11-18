# import the libraries
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# load the model and vectorizer
model = joblib.load('emailclassifier.pkl')
vectorizer = joblib.load('countvectorizer.pkl')

# define the request body
class EmailRequest(BaseModel):
    email: str

# define the home endpoint
app = FastAPI()
@app.get("/")
def home():
    return {"message": "Welcome to the Email Classifier API!"}

# define the predict endpoint
@app.post("/predict")
def predict(request: EmailRequest):
    # get the email from the request
    email = request.email

    # transform the email using the vectorizer
    email_vector = vectorizer.transform([email])

    # make the prediction using the model
    prediction = model.predict(email_vector)

    # return the prediction
    return {"prediction": prediction[0]}    
