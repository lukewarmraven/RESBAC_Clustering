import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

model = joblib.load('C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\regressorModel.pkl')

newData = pd.read_csv("ALGORITHMS/datasets/RFR/rfr615.csv")
#print(len(newData))
# preprocessing
X = newData.drop('priorityLevel',axis=1)
y = newData[['priorityLevel']]

# # trying manual weights
# weights = np.array([2,10,1,1,1,1,3,2])
"""
The code here uses RandomForest Regressor to predict continuous values of priority levels 0-3. By having a continuous value for output, it helps in ranking the prioritization of the user based on the algorithm. 

We are tasked to ensure that the main parameters for vulnerability of users have "weight". WE encoded the expert insights into the structure of the dataset itself using a scoring system. The scoring system for the main parameters uses 0-4 points, depending on the conditions like the ff. below:
Elderly:
    < 60 - 0
    60-70 - 1
    70-80 - 2
    80-90 - 3
    > 90 - 4

PWD: we intend to split each PWD types into Physical, Psych, and Sensory to not lose their individual importance by generalizing them into one parameter (PWD). The number of counted disabilities determine the points.
    Physical:
        Any - 0-4, 1 point for each, 4 max
    Sensory:
        Any - 0-4, 1 point for each, 4 max
    Psychological:
        Any - 0-4, 1 point for each, 4 max

Pregnant/Infant:
    Pregnant - 2
    Infant - 2
    Both - 4
    None - 0

Medically Dependent: The number of counted dependency determine the points.
    Any - 0-4, 1 point for each, 4 max

And these are the other parameters:
Has Guardian - 0-1
lcoationRiskLevel - 0-3

This script uses a dummy dataset. Our group plans to intentionally create a dataset with equal number of entries for each priority level so that the ML can effectively identify each properly by being able to see it in the training. 

We intend to create only the content for the parameters, but the result or answer for the priority level will come from experts and professionals.
"""
sampleData = pd.DataFrame([{
    'ElderlyScore':             0, #isElderly
    'PregnantOrInfantScore' :   2, #isPregnantOrInfant
    'PhysicalPWDScore':         0, #PhysicalScore
    'PsychPWDScore':            0, #PsychScore
    'SensoryPWDScore':          0, #SensoryScore
    'MedicallyDependentScore':  0, #isMedicallyDependent
    'hasGuardian':              0, #hasGuardian
    'locationRiskLevel':        0  #locationRiskLevel
}])

weights = np.array([1,4,1,1,1,1,1,1])

# Apply weights to new dataset
sampleData_weighted = sampleData * weights
samplePred = model.predict(sampleData_weighted)
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