import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv("C:\\Users\\bacqu\\Documents\\CAPSTONE PROJ\\priority_scores_balanced_rs1.csv")
# print(data.head())

X = data.drop('priorityLevel',axis=1)
y = data['priorityLevel']

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=15)

model = RandomForestClassifier(n_estimators=500, random_state=61)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,zero_division=0))
print(confusion_matrix(y_test,y_pred))

joblib.dump(model, 'rf_model.pkl')
