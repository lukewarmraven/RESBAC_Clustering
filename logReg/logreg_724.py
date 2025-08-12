import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib
import numpy as np

# data = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\dataset_random_state_45.csv")
# RS 1 IS 88% ACCURACY ######################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<
upData = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\priority_scores_balanced_rs45.csv")
# print(upData)
# print(data.isna().sum()) # checks how many null values are there for each column
# for data
# features = ['isElderly','isPregnantOrInfant','isPWD','isMedicallyDependent','needsEvacuationHelp','hasGuardian','locationRiskLevel']
# label = ['priorityLevel']
# X = data[features]
# y = data[label]

# for upData
upFeatures = ['isElderly','isPregnantOrInfant','isPWD','isMedicallyDependent','needsEvacuationHelp','hasGuardian','locationRiskLevel']
upLabel = ['priorityLevel']
upX = upData[upFeatures]
upy = upData[upLabel]

# splitting 
# data
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)
# upData
X_train,X_test,y_train,y_test = train_test_split(upX,upy,test_size=0.2,random_state=20)

model = LogisticRegression(max_iter=2000,solver='lbfgs')
model.fit(X_train,y_train.values.ravel())

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
# print("Classification Report: ", classification_report(y_test,y_pred,zero_division=0))
# the def below are based on how close to ZERO the values are
# precision - many false positive, guessed way too often, how many are the predicted labels that are correctly identified or predicted
# recall - many false negatives, missed actual instances, how many are the actual instances that are identified or predicted correctly
# f-score - tells if either precision or recall is balanced, low means bad prediction of classif
# support - number of actual items in the dataset of that classification

# print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

# Save model
joblib.dump(model, 'logreg_model.pkl')

# If you used a scaler (like StandardScaler), save it too
#joblib.dump(scaler, 'scaler.pkl')

# feature importance
coefficients = model.coef_
featureImportance = pd.DataFrame({
    'Feature': upFeatures,
    'Coefficient (importance)' : np.mean(np.abs(coefficients),axis=0)
})

featureImportance = featureImportance.sort_values(by='Coefficient (importance)',ascending=False)
# print(featureImportance)

# Make pandas show all columns and wider tables
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # No fixed width line wrapping
pd.set_option('display.max_colwidth', None)  # Show full content in each cell
# get prediction probability
y_proba = model.predict_proba(X_test)
# Combine feature values, predicted class, and probabilities into a DataFrame
results_df = X_test.copy()
results_df['Actual Priority'] = y_test.values
results_df['Predicted Priority'] = y_pred
results_df['Max Probability (%)'] = (y_proba.max(axis=1) * 100).round(2)

# Add columns for each class probability
class_labels = model.classes_  # e.g., array([0,1,2,3])
for idx, label in enumerate(class_labels):
    results_df[f'Prob_{label}'] = (y_proba[:, idx] * 100).round(2)

# Reset index for cleaner display
results_df = results_df.reset_index(drop=True)

# Show first 10 rows
print(results_df.head(10))