import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

model = joblib.load('C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\rf_model.pkl')

newData = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\priority_scores_balanced_rs45.csv")
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
# X.to_csv('rfPredicted.csv')
# print(y_pred)

sample1 = np.array([[
    1, #isElderly
    0, #isPregnantOrInfant
    0, #isPWD
    0, #isMedicallyDependent
    0, #needsEvacuationHelp
    1, #hasGuardian
    0  #locationRiskLevel
]])
predicted = model.predict(sample1)
print("Predicted Priority from Sample: ", predicted[0])