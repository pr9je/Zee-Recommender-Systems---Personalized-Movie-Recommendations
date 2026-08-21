import warnings
warnings.filterwarnings('ignore')

import os
import urllib.request
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = 'plotly_white'
# 'notebook' renderer embeds the chart library inline (once) so charts still render if this
# notebook is exported to HTML/PDF, not only when opened live in Jupyter/Colab/VS Code.
pio.renderers.default = 'notebook'

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option('display.max_columns', 50)

DATA_DIR = 'data'
FILES = ['ratings.dat', 'users.dat', 'movies.dat']
SOURCE = 'https://raw.githubusercontent.com/vandit15/Movielens-Data/master/ml-1m/{}'

os.makedirs(DATA_DIR, exist_ok=True)
for fname in FILES:
  fpath = os.path.join(DATA_DIR, fname)
  if not os.path.exists(fpath):
    print(f'Downloading {fname}...')
    urllib.request.urlretrieve(SOURCE.format(fname), fpath)
print("Data files ready:", os.listdir(DATA_DIR))


# Data Loading and Merging
ratings = pd.read_csv(f'{DATA_DIR}/ratings.dat', sep='::', engine='python',
                       names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')
users = pd.read_csv(f'{DATA_DIR}/users.dat', sep='::', engine='python',
                     names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')
movies = pd.read_csv(f'{DATA_DIR}/movies.dat', sep='::', engine='python',
                      names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

print(f"ratings: {ratings.shape}, users: {users.shape}, movies: {movies.shape}")
ratings.head()
users.head()
movies.head()

# Merge into a single workging dataframe: one row per (user, moive) rating, enriched with user demographics and moive metadata.
df = ratings.merge(users, on='UserID').merge(movies, on='MovieID')
print("Merged dataframe shape: ", df.shape)
df.head()
