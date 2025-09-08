import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import joblib

model = joblib.load('ALGORITHMS\\Multinomial LogReg\\multinomialModel.pkl')

newData = pd.read_csv('ALGORITHMS/datasets/dataset615.csv')

X = newData.drop('priorityLevel',axis=1)
y = newData[['priorityLevel']]

y_pred = model.predict(X)

# print("Accuracy:", accuracy_score(y,y_pred))
# print("Classification Report: \n",classification_report(y,y_pred))

sampleData = pd.DataFrame([{
    'ElderlyScore':             0, #isElderly
    'PregnantOrInfantScore' :   3, #isPregnantOrInfant
    'PhysicalPWDScore':         0, #PhysicalScore
    'PsychPWDScore':            0, #PsychScore
    'SensoryPWDScore':          0, #SensoryScore
    'MedicallyDependentScore':  0, #isMedicallyDependent
    'needsEvacuationHelp':      1, #needsEvacuationHelp
    'hasGuardian':              0, #hasGuardian
    'locationRiskLevel':        1  #locationRiskLevel
}])
samplePred = model.predict(sampleData)
print("Priority Level: ", samplePred[0])

# feature importance
coefficients = model.coef_
features = sampleData.columns.tolist()
featureImportance = pd.DataFrame({
    'Feature': features,
    'Importance' : np.mean(np.abs(coefficients),axis=0)
})
featureImportance = featureImportance.sort_values(by='Importance',ascending=False)
print(featureImportance)