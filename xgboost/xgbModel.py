import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

model = xgb.Booster()
model.load_model("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\xgbModel.json")

# ===========================
# LOAD DATA
# ===========================
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.read_csv("ALGORITHMS/datasets/RFR/rfr615.csv")

X = data.drop('priorityLevel', axis=1)
y = data['priorityLevel']

feature_weights = {
    "ElderlyScore": 1,
    "PregnantOrInfantScore": 1,
    "PhysicalPWDScore": 1,
    "PsychPWDScore": 1,
    "SensoryPWDScore": 1,
    "MedicallyDependentScore": 1,
    "hasGuardian": 1,          
    "locationRiskLevel": 1
}
# Convert dict → list in the same order as X.columns
weights_list = [feature_weights[col] for col in X.columns]

# ===========================
# SAMPLE PREDICTION
# ===========================
sampleData = pd.DataFrame([{
    'ElderlyScore':             0,
    'PregnantOrInfantScore':    4,
    'PhysicalPWDScore':         0,
    'PsychPWDScore':            0,
    'SensoryPWDScore':          0,
    'MedicallyDependentScore':  1,
    'hasGuardian':              0,
    'locationRiskLevel':        3
}])

dsample = xgb.DMatrix(sampleData, feature_weights=weights_list)
samplePred = model.predict(dsample)
print("Predicted Priority from Sample:", samplePred[0])

# ===========================
# FEATURE IMPORTANCE
# ===========================
importance = model.get_score(importance_type="weight")
importance_df = pd.DataFrame({
    "Feature": list(importance.keys()),
    "Importance": list(importance.values())
}).sort_values(by="Importance", ascending=False)

print(importance_df)

# SHAP EXPLANATION
explainer = shap.Explainer(model)
shap_values = explainer(sampleData)

# # Local effect (for one row)
shap.plots.waterfall(shap_values[0])
# # Global summary
# shap.plots.beeswarm(shap_values)

# # cross-validate r2
# kf = KFold(n_splits=5, shuffle=True, random_state=45)
# r2_scores = []
# for train_idx, test_idx in kf.split(X):
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=weights_list)
#     dtest = xgb.DMatrix(X_test, label=y_test, feature_weights=weights_list)

#     model = xgb.train(params, dtrain, num_boost_round=100)
#     y_pred = model.predict(dtest)

#     r2_scores.append(r2_score(y_test, y_pred))

# print("Cross-validated R²:", np.mean(r2_scores), "+/-", np.std(r2_scores))
# print("MSE:", mean_squared_error(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))