import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("sample_evacuation_dataset_with_pregnancy.csv")

y = df["Can_Self_Evacuate"]
x = df.drop("Can_Self_Evacuate",axis=1)
#x["Disaster Alert"] = 1
# print(x)

# TRAINING AND TESTING
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)

# PREDICTING
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred), "\n")

for feature, coef in zip(x.columns, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")

results = x_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["% Can Ev"] = y_proba[:,1] * 100
results["% Can't Ev"] = y_proba[:,0] * 100

print(results)


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Confusion matrix heatmap
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()


