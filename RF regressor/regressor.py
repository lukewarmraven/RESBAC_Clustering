import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("ALGORITHMS/datasets/dataset45.csv")
# print(data.head())

X = data.drop('priorityLevel',axis=1)
y = data['priorityLevel']

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=15)

model = RandomForestRegressor(n_estimators=100, random_state=61)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# print(accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred,zero_division=0))
# print(confusion_matrix(y_test,y_pred))

joblib.dump(model, 'regressorModel.pkl')
