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
