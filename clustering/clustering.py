#from data import normalData
#from closeData import dataCoords
from pprint import pprint
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import gmplot
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# changes directory to current dir to create mapHTML file there
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print(os.getcwd())

def cluster(k=10,risk_num=1):
    #NORMAL DATA 
    #pprint(normalData[0], sort_dicts=False)
    #normal = pd.DataFrame(normalData)
    #locNormal = normal[["lat","lng"]]

    location_risk = risk_num

    # data DATA
    #dataCoords_dicts = [{"lat": list(coord)[1], "lng": list(coord)[0]} for coord in dataCoords]
    #data = pd.DataFrame(dataCoords_dicts)
    data = pd.read_csv("closeClustered.csv")
    # multiple parameters
    #features = data[['lat', 'lng', 'evac_capability', 'risk_location']]
    
    # for ALL button
    if location_risk != 0:
        data = data[data['risk_location'] == location_risk]

    #data = data[data['risk_location'] == location_risk]
    # geolocation based only
    features = data[['lat','lng']]
    #print(data)
    
    # Check if there are enough data points to form k clusters
    if len(features) < k:
        print(f"[WARNING] Only {len(features)} points available — reducing k from {k} to {len(features)}")
        k = len(features)

    if k == 0:
        k = 1
    
    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    #print(normalized_features)

    # training model to identify
    model = KMeans(n_clusters=k)
    #model = KMeans(n_clusters=k,random_state=42)
    data['result'] = model.fit_predict(normalized_features)

    #normal['result'] = y_kmeans
    #data['result'] = y_kmeans

    # save file to csv for mapPlotting
    #normal.to_csv("normalClustered.csv",index=False)
    #data.to_csv("dataClustered.csv",index=False)

    # check plotting here
    # plt.scatter(data["lng"],data["lat"],c=data['result'])
    # plt.show()

    # MAPPING
    # data for plotting
    #data = pd.read_csv('dataClustered.csv') # can be changed to what's easier
    # can just get the value of dataCoords

    load_dotenv()
    # changed from data to data variable
    api_key = os.getenv('API_KEY')
    lat = data['lat'].tolist()
    lng = data['lng'].tolist()
    result = data['result'].tolist()

    # add colors if u add another cluster
    # CANNOT EXCEED 25 CLUSTERS
    colors = [
        'red', 'blue', 'green', 'violet', 'yellow', 'orange',
        'pink', 'white', 'black', 'brown', 'cyan', 'magenta',
        'lime', 'indigo', 'gray', 'olive', 'teal', 'maroon',
        'gold', 'coral', 'turquoise', 'navy', 'purple', 'salmon'
    ]
    clusterColor = [colors[cluster] for cluster in result]

    # creates mapping
    gmap1 = gmplot.GoogleMapPlotter(14.660915632697726, 121.10051851107902,15,apikey=api_key)
    #gmap1.scatter(lat,lng,'#ff0000',size=10,marker='o')

    for latPoint,lngPoint, color in zip(lat,lng,clusterColor):
        gmap1.marker(latPoint,lngPoint,color=color)

    # creates new html file to show whole mapping
    gmap1.draw('static/newMap.html')

#cluster()