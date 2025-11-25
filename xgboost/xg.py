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
data = pd.read_csv("..\\datasets\\mock\\ds45 copy.csv")

X = data.drop('priorityLevel', axis=1)
y = data['priorityLevel']

# ===========================
# FEATURE WEIGHTING (for DMatrix)
# feature_weights = {
#     "ElderlyScore": 1,
#     "PregnantOrInfantScore": 1,
#     "PhysicalPWDScore": 1,
#     "PsychPWDScore": 1,
#     "SensoryPWDScore": 1,
#     "MedicallyDependentScore": 1,
#     # "hasGuardian": 3,          
#     "locationRiskLevel": 1
# }

# # Convert dict → list in the same order as X.columns
# weights_list = [feature_weights[col] for col in X.columns]

# ===========================
# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# Compute class frequencies
classes, counts = np.unique(y_train, return_counts=True)
total = len(y_train)

# Weight = total_samples / (num_classes * class_count)
# -> rarer classes get larger weights
class_weights = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}

# Assign weight to each training row
train_weights = y_train.map(class_weights).values
test_weights = y_test.map(class_weights).values

# Create DMatrix with weights
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)

# ===========================
# MODEL (XGBoost train API)
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "lambda": 1,          # L2 regularization
    "alpha": 0.5,         # L1 regularization
    "seed": 45
}

model = xgb.train(params, dtrain, num_boost_round=500,evals=[(dtest, "test")], early_stopping_rounds=30, verbose_eval=False)
y_pred = model.predict(dtest)

# ==========================
# cross-validate r2 (with early stopping, same as training)
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=45)
r2_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # keep row weights instead of feature_weights in CV
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}
    train_weights = y_train.map(class_weights).values
    test_weights = y_test.map(class_weights).values

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=test_weights)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,            # match training
        evals=[(dtest, "test")],
        early_stopping_rounds=30,       # match training
        verbose_eval=False
    )

    y_pred = model.predict(dtest)
    r2_scores.append(r2_score(y_test, y_pred))

print("Cross-validated R²:", np.mean(r2_scores), "+/-", np.std(r2_scores))

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
# model.save_model("../../prio/xgb.json")
# model.save_model("../../xgb.json")

# ===========================
# SAMPLE PREDICTION
# ===========================
sampleData = pd.DataFrame([{
        'ElderlyScore':             2,
        'PregnantOrInfantScore':    0,
        'PhysicalPWDScore':         0,
        'PsychPWDScore':            0,
        'SensoryPWDScore':          0,
        'MedicallyDependentScore':  0,
        'locationRiskLevel':        3
}])

dsample = xgb.DMatrix(sampleData)
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
# shap_values = explainer(dsample)
shap_values = explainer(dsample)

# # Local effect (for one row)
# shap.plots.waterfall(shap_values[0])
# print(X['hasGuardian'].corr(y))
# print(X['locationRiskLevel'].corr(y))
# # Global summary
# shap.plots.beeswarm(shap_values)