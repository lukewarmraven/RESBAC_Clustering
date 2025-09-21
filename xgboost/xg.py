import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

# ===========================
# LOAD DATA
# ===========================
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.read_csv("..\\datasets\\RFR\\rfr45.csv")

X = data.drop('priorityLevel', axis=1)
y = data['priorityLevel']

# ===========================
# FEATURE WEIGHTING (for DMatrix)
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

# Convert dict → list in the same order as X.columns
weights_list = [feature_weights[col] for col in X.columns]

# ===========================
# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# Create DMatrix with feature weights
dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=weights_list)
dtest = xgb.DMatrix(X_test, label=y_test, feature_weights=weights_list)

# ===========================
# MODEL (XGBoost train API)
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1,          # L2 regularization
    "alpha": 0.5,         # L1 regularization
    "seed": 45
}

model = xgb.train(params, dtrain, num_boost_round=50,evals=[(dtest, "test")], early_stopping_rounds=20, verbose_eval=False)
y_pred = model.predict(dtest)

# ==========================
# cross-validate r2
kf = KFold(n_splits=5, shuffle=True, random_state=45)
r2_scores = []
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_weights=weights_list)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_weights=weights_list)

    model = xgb.train(params, dtrain, num_boost_round=50)
    y_pred = model.predict(dtest)

    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated R²:", np.mean(r2_scores), "+/-", np.std(r2_scores))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
model.save_model("xgbModel.json")

# ===========================
# SAMPLE PREDICTION
# ===========================
sampleData = pd.DataFrame([{
        'ElderlyScore':             1,
        'PregnantOrInfantScore':    2,
        'PhysicalPWDScore':         0,
        'PsychPWDScore':            0,
        'SensoryPWDScore':          0,
        'MedicallyDependentScore':  0,
        'hasGuardian':              1,
        'locationRiskLevel':        3
}])

dsample = xgb.DMatrix(sampleData, feature_weights=weights_list)
samplePred = model.predict(dsample)
print("Predicted Priority from Sample:", samplePred[0])

# ===========================
# FEATURE IMPORTANCE
# ===========================
importance = model.get_score(importance_type="gain")
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
print(X['hasGuardian'].corr(y))
print(X['locationRiskLevel'].corr(y))
# # Global summary
# shap.plots.beeswarm(shap_values)

# ===========================
# CROSS-VALIDATION (R2)
# ===========================
# dall = xgb.DMatrix(X, label=y, feature_weights=weights_list)

# cv_results = xgb.cv(
#     params,
#     dall,
#     num_boost_round=100,
#     nfold=5,                        # 5-fold CV
#     metrics=["rmse", "mae"],                   # ask for R²
#     seed=45,
#     as_pandas=True
# )

# print("Mean CV MAE:", cv_results["test-mae-mean"].iloc[-1])
# print("Std CV MAE:", cv_results["test-mae-std"].iloc[-1])
# print("Mean CV RMSE:", cv_results["test-rmse-mean"].iloc[-1])
# print("Std CV RMSE:", cv_results["test-rmse-std"].iloc[-1])