import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

model = joblib.load('C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\regressorModel.pkl')

newData = pd.read_csv("ALGORITHMS/datasets/dataset615.csv")
#print(len(newData))
# preprocessing
X = newData.drop('priorityLevel',axis=1)
y = newData[['priorityLevel']]

y_pred = model.predict(X)

sampleData = pd.DataFrame([{
    'ElderlyScore':             0, #isElderly
    'PregnantOrInfantScore' :   0, #isPregnantOrInfant
    'PhysicalPWDScore':         2, #PhysicalScore
    'PsychPWDScore':            1, #PsychScore
    'SensoryPWDScore':          1, #SensoryScore
    'MedicallyDependentScore':  1, #isMedicallyDependent
    'needsEvacuationHelp':      1, #needsEvacuationHelp
    'hasGuardian':              0, #hasGuardian
    'locationRiskLevel':        3  #locationRiskLevel
}])
samplePred = model.predict(sampleData)
print("Predicted Priority from Sample: ", samplePred[0])

# FEATURE IMPORTANCE
feature_names = sampleData.columns.tolist()
feature_importances = model.feature_importances_
# Create a DataFrame to display the feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Print the sorted feature importance
print(feature_importance_df)