import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib


data = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\dataset_random_state_54.csv")
#print(data.head())
#print(data.isna().sum()) # checks how many null values are there for each column
features = ['isElderly','isPregnantOrInfant','isPWD','isMedicallyDependent','needsEvacuationHelp','hasGuardian','locationRiskLevel']
label = ['priorityLevel']
X = data[features]
y = data[label]

# splitting 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)
model = LogisticRegression(max_iter=1000,multi_class='multinomial',solver='lbfgs')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
print("Classification Report: ", classification_report(y_test,y_pred,zero_division=0))
# the def below are based on how close to ZERO the values are
# precision - many false positive, guessed way too often, how many are the predicted labels that are correctly identified or predicted
# recall - many false negatives, missed actual instances, how many are the actual instances that are identified or predicted correctly
# f-score - tells if either precision or recall is balanced, low means bad prediction of classif
# support - number of actual items in the dataset of that classification
print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

# Save model
joblib.dump(model, 'logreg_model.pkl')

# If you used a scaler (like StandardScaler), save it too
#joblib.dump(scaler, 'scaler.pkl')