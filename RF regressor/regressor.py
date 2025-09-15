import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.read_csv("ALGORITHMS/datasets/RFR/rfr45.csv")
# print(data.head())

# Example weights (aligned with your columns order)
# weights = np.array([1,3,1,1,1,1,1,1])

# Apply feature weighting by scaling columns
X = data.drop('priorityLevel',axis=1)
y = data['priorityLevel']

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=15)

sample_weights = np.ones(len(y))  

sample_weights[X["hasGuardian"] == 0] *= 3              # No guardian → double weight
sample_weights[X["PregnantOrInfantScore"] > 0] *= 3     # Pregnant/Infant → triple weight
# sample_weights[X["locationRiskLevel"] == 3] *= 2        # Highest risk → double weight

model = RandomForestRegressor(n_estimators=100, random_state=45,max_depth=5)
model.fit(X_train,y_train,sample_weight=sample_weights[X_train.index])

y_pred = model.predict(X_test)

scores = cross_val_score(model, X, y, cv=5, scoring='r2',fit_params={'sample_weight': sample_weights})
print("Cross-validated R2:", scores.mean(), scores.std())
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, 'regressorModel.pkl')

sampleData = pd.DataFrame([{
    'ElderlyScore':             0, #isElderly
    'PregnantOrInfantScore' :   2, #isPregnantOrInfant
    'PhysicalPWDScore':         0, #PhysicalScore
    'PsychPWDScore':            0, #PsychScore
    'SensoryPWDScore':          0, #SensoryScore
    'MedicallyDependentScore':  0, #isMedicallyDependent
    'hasGuardian':              0, #hasGuardian
    'locationRiskLevel':        3  #locationRiskLevel
}])
sampleData_weighted = sampleData
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
