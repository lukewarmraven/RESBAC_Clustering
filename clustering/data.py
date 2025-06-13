import pandas as pd

# make different datasets to train the clustering according to scenario
# 1 - 15 loc: normall distributed coords
# 2 - 5 loc: very close to each other
# 3 - 20 loc: with outliers
locationList = [
    {"lat":14.656709,"lng":121.097099},
    {"lat":14.658370,"lng":121.093344},
    {"lat": 14.657768, "lng": 121.101369},
    {"lat": 14.654903, "lng": 121.094589},
    {"lat": 14.655858, "lng": 121.090533},
    {"lat": 14.658308, "lng": 121.094696},
    {"lat": 14.655297, "lng": 121.097421},
    {"lat": 14.656024, "lng": 121.095790},
    {"lat": 14.656211, "lng": 121.101326},
    {"lat": 14.663975, "lng": 121.095211},
    {"lat": 14.653512, "lng": 121.092228},
    {"lat": 14.654812, "lng": 121.090606},
    {"lat": 14.654060, "lng": 121.089691},
    {"lat": 14.655771, "lng": 121.093123},
    {"lat": 14.654860, "lng": 121.102301},
    {"lat": 14.652379, "lng": 121.100323},
    {"lat": 14.656468, "lng": 121.094626},
    {"lat": 14.657372, "lng": 121.097072},
    {"lat": 14.654352, "lng": 121.099481},
    {"lat": 14.652702, "lng": 121.099183},
    {"lat": 14.655260, "lng": 121.101168}
]

normalData = [
    {"name":"Raven Quinto",
     "age":70,
     "contact":"09123456789",
     "vulnerability":["Senior Citizen","PWD"],
     **locationList[0]},
      {
        "name": "Juan Dela Cruz",
        "age": 65,
        "contact": "09111111111",
        "vulnerability": ["Senior Citizen"],
        **locationList[1]
    },
    {
        "name": "Maria Santos",
        "age": 29,
        "contact": "09222222222",
        "vulnerability": ["PWD"],
        **locationList[2]
    },
    {
        "name": "Jose Rizal",
        "age": 12,
        "contact": "09333333333",
        "vulnerability": ["Minor", "PWD"],
        **locationList[3]
    },
    {
        "name": "Ana Lopez",
        "age": 80,
        "contact": "09444444444",
        "vulnerability": ["Senior Citizen", "PWD", "Bedridden"],
        **locationList[4]
    },
    {
        "name": "Mark Reyes",
        "age": 35,
        "contact": "09555555555",
        "vulnerability": ["PWD"],
        **locationList[5]
    },
    {
        "name": "Liza Cruz",
        "age": 8,
        "contact": "09666666666",
        "vulnerability": ["Minor"],
        **locationList[6]
    },
    {
        "name": "Carlos Tan",
        "age": 74,
        "contact": "09777777777",
        "vulnerability": ["Senior Citizen"],
        **locationList[7]
    },
    {
        "name": "Nina Garcia",
        "age": 67,
        "contact": "09888888888",
        "vulnerability": ["Senior Citizen", "PWD"],
        **locationList[8]
    },
    {
        "name": "Pedro Santos",
        "age": 45,
        "contact": "09999999999",
        "vulnerability": ["PWD"],
        **locationList[9]
    },
    {
        "name": "Julia Ramos",
        "age": 10,
        "contact": "09101234567",
        "vulnerability": ["Minor", "PWD"],
        **locationList[10]
    },
    {
        "name": "Carlos Reyes",
        "age": 34,
        "contact": "09312345678",
        "vulnerability": ["PWD"],
        **locationList[11]
    },
    {
        "name": "Liza Fernandez",
        "age": 45,
        "contact": "09122334455",
        "vulnerability": ["Senior Citizen"],
        **locationList[12]
    },
    {
        "name": "Mark Tan",
        "age": 28,
        "contact": "09211223344",
        "vulnerability": ["Senior Citizen", "PWD"],
        **locationList[13]
    },
    {
        "name": "Ana Garcia",
        "age": 60,
        "contact": "09159873642",
        "vulnerability": ["Senior Citizen"],
        **locationList[14]
    },
    {
        "name": "Oscar Lim",
        "age": 50,
        "contact": "09345678901",
        "vulnerability": ["Senior Citizen", "PWD"],
        **locationList[15]
    },
    {
        "name": "Rachel Mendoza",
        "age": 32,
        "contact": "09123456788",
        "vulnerability": ["PWD"],
        **locationList[16]
    },
    {
        "name": "Felipe Santiago",
        "age": 40,
        "contact": "09223456789",
        "vulnerability": ["Senior Citizen"],
        **locationList[17]
    },
    {
        "name": "Julie Reyes",
        "age": 55,
        "contact": "09123456777",
        "vulnerability": ["PWD", "Senior Citizen"],
        **locationList[18]
    },
    {
        "name": "Eliot Cruz",
        "age": 22,
        "contact": "09234567890",
        "vulnerability": ["PWD"],
        **locationList[19]
    },
    {
        "name": "Veronica Perez",
        "age": 38,
        "contact": "09323456789",
        "vulnerability": ["Senior Citizen"],
        **locationList[20]
    }
]


closeData = []
outlierData = []
