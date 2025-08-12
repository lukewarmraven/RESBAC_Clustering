"""
The code here uses multinomial logistic regression to accommodate multi-classed priority levels for classifying or predicting disaster rescue priority level. 
We are tasked to ensure that the main parameters for vulnerability of users provide "weight". To do this using the LR, we used a scoring system for the main parameters, 0-4 points, depending on the conditions like the ff:
Elderly:
    < 60 - 0
    60-70 - 1
    70-80 - 2
    80-90 - 3
    > 90 - 4

PWD: we intend to split each PWD types in the parameters to not lose their individual importance by generalizing them into one parameter (PWD)
    Physical - 0-4
    Sensory - 0-4
    Psychological - 0-4

Pregnant/Infant:
    Pregnant - 0-1
    Infant - 0-1
    Both - 0-2

Medically Dependent:
    Any - 1 point for each

And these are the other parameters:
Evacuation Capability - 0-1 
Has Guardian - 0-1
lcoationRiskLevel - 0-3

This script uses a dummy dataset. Our group plans to intentionally create a dataset with equal number of entries for each priority level so that the ML can effectively identify each properly. 

We intent to create only the content for the parameters, but the result or answer for the priority level will come from experts and professionals.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import joblib
import numpy as np
import os

data = pd.read_csv("ALGORITHMS/datasets/dataset45.csv")
# need exact names with the columns in the dataset
features = ['ElderlyScore','PregnantOrInfantScore','PhysicalPWDScore','PsychPWDScore','SensoryPWDScore','MedicallyDependentScore','needsEvacuationHelp','hasGuardian','locationRiskLevel']
label = 'priorityLevel'
X = data[features]
y = data[label]

# splitting the dataset to train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)

model = LogisticRegression(max_iter=2000,solver='lbfgs')
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report: \n", classification_report(y_test,y_pred,zero_division=0))
"""the def below are based on how close to ZERO the values are for classif report:
precision - many false positive, guessed way too often, how many are the predicted labels that are correctly identified or predicted
recall - many false negatives, missed actual instances, how many are the actual instances that are identified or predicted correctly
f-score - tells if either precision or recall is balanced, low means bad prediction of classif
support - number of actual items in the dataset of that classification"""

# see results
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted':y_pred    
})
# get probability percentage
y_proba = model.predict_proba(X_test)
predicted_prob = y_proba.max(axis=1) * 100 
results['Probability %'] = predicted_prob
# getting the parameters 
results['Parameters'] = X_test.apply(lambda row: row.tolist(),axis=1)
print(results)

# Save model
path = os.path.dirname(__file__)
joblib.dump(model, os.path.join(path,'multinomialModel.pkl'))
