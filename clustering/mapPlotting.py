import pandas as pd
import gmplot
#from clustering import normal
# my goal is to not import from the clustering file and instead access the csv produced from there
from dotenv import load_dotenv
import os
from io import StringIO

load_dotenv()

# changes directory to current dir to create mapHTML file there
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print(os.getcwd())

data = pd.read_csv('closeClustered.csv')

# trying to map them in google maps
import gmplot

api_key = os.getenv('API_KEY')
lat = data['lat'].tolist()
lng = data['lng'].tolist()
result = data['result'].tolist()

# add colors if u add another cluster
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