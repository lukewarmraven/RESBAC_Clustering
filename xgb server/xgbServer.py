import xgboost as xgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model = xgb.Booster()
model.load_model('../xgboost/xgbModel.json')

app = FastAPI()

""" for the new feature weights to reflect,
the model must be updated/trained with the new feature weights
and put the feature weights here below VVVV
"""
feature_weights = {
    "ElderlyScore": 1,
    "PregnantOrInfantScore": 1,
    "PhysicalPWDScore": 1,
    "PsychPWDScore": 1,
    "SensoryPWDScore": 1,
    "MedicallyDependentScore": 1,
    "hasGuardian": 3,          
    "locationRiskLevel": 1
}
weights_list = list(feature_weights.values())


class InputData(BaseModel):
    values: dict

@app.post('/predict')
def predict(data:InputData):
    input_list = [data.values[feat] for feat in feature_weights.keys()]
    arr = np.array(input_list).reshape(1, -1)

    dmdata = xgb.DMatrix(
        arr,
        feature_names=list(feature_weights.keys()),
        )
    
    prediction = model.predict(dmdata)
    return {
        "prediction":prediction.tolist()
    }

if __name__ == "__main__":
    uvicorn.run("xgbServer:app",host='0.0.0.0',port=8000,reload=True)



