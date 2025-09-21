import requests
import xgboost as xgb

url = 'http://127.0.0.1:8000/predict'

data = {
    "values":
    {
        'ElderlyScore':             1,
        'PregnantOrInfantScore':    2,
        'PhysicalPWDScore':         0,
        'PsychPWDScore':            0,
        'SensoryPWDScore':          0,
        'MedicallyDependentScore':  0,
        'hasGuardian':              0,
        'locationRiskLevel':        3
    }
}

response = requests.post(url,json=data)
if response == 200:
    print("Prediction",response.json())
else: 
    print("Error: ", response.status_code, response.text)