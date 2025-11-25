import xgboost as x
import pandas as pd
import numpy as np

# slot the model path here
model = x.Booster()
model.load_model('../xgb.json')

# slot the dataset path here
data = pd.read_csv('../../datasets/newD1.csv')
pd.set_option('display.max_columns', None)


# data cleaning
# data[data['PhysicalPWDScore'].isnull()] = 0
# threes = data[data['PhysicalPWDScore'].isnull()]
# print(threes)
# md = data[data['MedDepScore'].isna()]
# print(md)
# ps = data[data['PsychPWDScore'] == 3]
# print(ps)

# for training the whole dataset
def train_all(dataset):
    dsample = x.DMatrix(dataset)
    prediction = model.predict(dsample)
    data["prio"] = np.round(prediction).astype(int)
    data.to_csv('withPrio.csv')
    print(prediction)

# for one entry prioritization 
def train_one(data):
    dsample = x.DMatrix(data)
    prediction = model.predict(dsample)
    print(prediction)

def prioStats():
    prioData = pd.read_csv('withPrio.csv')
    unique = prioData['prio'].value_counts()
    print(unique)
    print("Total of prio: ", unique.sum())
    
    total = prioData.count()
    print("Total of the dataset: \n", total)



# choose from the functions here
# print("""
# Choose one of the following options:
# (1) Train the whole dataset
# (2) Train the sample data
# >>>
# """)

userchoice = input("""
Choose one of the following options:
(1) Train the whole dataset
(2) Train the sample data
(3) Statistics of results
>>>  """)

if (userchoice == "1"):
    print("Training all data...")
    train_all(data)
elif(userchoice == "2"):
    print("Training sample data...")
    sampleData = pd.DataFrame([{
        'ElderlyScore':             2,
        'PregnantOrInfantScore':    0,
        'PhysicalPWDScore':         0,
        'PsychPWDScore':            0,
        'SensoryPWDScore':          0,
        'MedicallyDependentScore':  0,
        'locationRiskLevel':        3
    }])
    train_one(sampleData)
elif(userchoice == "3"):
    print("Count of priority levels:")
    prioStats()
else:
    print("Option not valid. Processed terminated. Run again to restart.")