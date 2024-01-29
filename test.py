import pandas as pd 
import joblib
from io import BytesIO

import requests # or https

# e.g. a file call stopwords saved by joblib
# https://github.com/Proteusiq/hisia/v1.0.1/hisia/models/data/stops.pkl

# change github.com to raw.githubusercontent.com

#URI = "http://github.com\\TylerJamesMalloy\\88308\\blob\\main\\Data\\NivData.pkl"

link = 'https://github.com/TylerJamesMalloy/88308/blob/main/Data/NivData.pkl'

from io import BytesIO
import pickle
import requests
mLink = 'https://github.com/TylerJamesMalloy/88308/blob/main/Data/NivData.pkl?raw=true'
mfile = BytesIO(requests.get(mLink).content)
data = pickle.load(mfile)

print(data)