import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

model = joblib.load('C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\logreg_model.pkl')

newData = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\priority_scores_balanced_rs615.csv")
#print(len(newData))
# preprocessing
X = newData.drop('priorityLevel',axis=1)
y = newData[['priorityLevel']]

y_pred = model.predict(X)

# print("Accuracy Score: ", accuracy_score(y,y_pred))
# print("Classification Report: ",classification_report(y,y_pred))
# print("Confusion: \n", confusion_matrix(y,y_pred))
# X['priorityPrediction'] = y_pred
# print(X)
# X.to_csv('logregPredicted.csv')
# print(y_pred)

sample1 = np.array([[
    1, #isElderly
    0, #isPregnantOrInfant
    1, #isPWD
    1, #isMedicallyDependent
    0, #needsEvacuationHelp
    0, #hasGuardian
    1  #locationRiskLevel
]])
predicted = model.predict(sample1)
print("Predicted Priority from Sample: ", predicted[0])

# feature importance
coefficients = model.coef_
upFeatures = ['isElderly','isPregnantOrInfant','isPWD','isMedicallyDependent','needsEvacuationHelp','hasGuardian','locationRiskLevel']

featureImportance = pd.DataFrame({
    'Feature': upFeatures,
    'Coefficient (importance)' : np.mean(np.abs(coefficients),axis=0)
})

featureImportance = featureImportance.sort_values(by='Coefficient (importance)',ascending=False)
# print(featureImportance)