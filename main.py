
# Data Handling
import logging
import pickle
import numpy as np
from pydantic import BaseModel

# Server
import uvicorn
from fastapi import FastAPI, Body

# Modeling
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# Initialize files

rf = pickle.load(open('dataset/rf.pickle', 'rb'))
encoderLabel = pickle.load(open('dataset/encoderLa.pickle', 'rb'))
features = pickle.load(open('dataset/selected_feat.pickle', 'rb'))

class Data(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    compactness_mean: float
    concavity_mean: float
    perimeter_se: float
    area_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    compactness_worst: float
    concavity_worst: float
    
        
@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = np.array([data_dict[feature] for feature in features])
        print(to_predict.reshape(1, -1).shape)

        # Create and return prediction
        
        prediction_rf = rf.predict(to_predict.reshape(1, -1))

        rf_proba = max(rf.predict_proba(to_predict.reshape(1, -1))[0])
        
        prediction = {
            
            'Prediction': "{} ({:.2f}%)".format(encoderLabel.inverse_transform(prediction_rf)[0],rf_proba*100),
        }
        return prediction
    
    except:
        my_logger.error("Something went wrong!")
        prediction_error = {
            
            'Prediction': 'error',
        }
        return prediction_error

@app.get("/example")
async def input_example():
    example = {
        "radius_mean": 7.760,
        "texture_mean": 24.54,
        "perimeter_mean": 47.92,
        "area_mean": 181.0,
        "compactness_mean": 0.05263,
        "concavity_mean": 0.04362,
        "perimeter_se": 0.0,
        "area_se": 0.0,
        "radius_worst": 0.1587,
        "texture_worst": 0.05884,
        "perimeter_worst": 0.3857,
        "area_worst": 1.428,
        "compactness_worst": 2.548,
        "concavity_worst": 19.15,
    }
    return example


