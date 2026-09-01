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

# EDA
# Shape, structure & datatypes
df.info()

df.describe(include='all').T

# Data consistency checks
valid_ages = {1, 18, 25, 35, 45, 50, 56}

print("Fully duplicated rows:", df.duplicated().sum())
print("Duplicated (UserID, MovieID) pairs: ", df.duplicated(subset=['UserID', 'MovieID']).sum(), "<- same user rating the same movie twice, would need de-duping if > 0")
print("Rating values present: ", sorted(df['Rating'].unique().tolist()), "-> all within 1-5:", df['Rating'].between(1,5).all())
print("Age codes present: ", sorted(df['Age'].unique().tolist()),"-> all within documented set:", set(df['Age'].unique().tolist()).issubset(valid_ages))
print("Occupation codes present: ", sorted(df['Occupation'].unique().tolist()), "-> all within 0-20:", df['Occupation'].between(0, 20).all())
print("Gender values present:", sorted(df['Gender'].unique().tolist()))

bad_titles = movies.loc[movies['Title'].str.extract(r'\((\d{4})\)$')[0].isna(), 'Title'].tolist()
print(f"Titles that don't match 'Title (YYYY)':", len(bad_titles), bad_titles)

ts = df['Timestamp']
print(f"Timestamps all positive: {(ts > 0).all()}  |  range: {pd.to_datetime(ts.min(), unit='s').date()} to {pd.to_datetime(ts.max(), unit='s').date()}")
# Missing values
print("Missing values per column:")
print(df.isna().sum())

# Scale & Sparsity
n_users, n_movies, n_ratings = df['UserID'].nunique(), df['MovieID'].nunique(), len(df)

print(f"Unique Users: {n_users}")
print(f'Unique Movies: {n_movies} note: fewer than the 3,883 in movies.dat - not every catalog title has been rated')
print(f'Total ratings: {n_ratings}')
print(f'Ratings per user -> min: {df.groupby('UserID').size().min()}, mean: {df.groupby('UserID').size().mean():.1f}, max: {df.groupby("UserID").size().max()}')
print(f'Ratings per movie -> min: {df.groupby('MovieID').size().min()}, mean: {df.groupby('MovieID').size().mean():.1f}, max: {df.groupby('MovieID').size().max()}')

sparsity = 1 - n_ratings / (n_users * n_movies)
print(f'\nUser-item matrix sparsity: {sparsity:.2%} of all possible (user, movie) paris are unrated')

# Rating distribution & User activity.
import plotly.express as px
import plotly.io as pio

rating_counts = df['Rating'].value_counts().sort_index().reset_index()
rating_counts.columns = ['Rating', 'Count']

fig = px.bar(
    rating_counts,
    x='Rating',
    y='Count',
    text='Count',
    title='Distribution of Movie Ratings',
    color_discrete_sequence=['#4C72B0']
)

fig.update_traces(
    texttemplate='%{text:,}',
    textposition='outside'
)

fig.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title='Count',
    bargap=0.3
)

pio.renderers.default = 'colab'
fig.show()

import plotly.express as px

ratings_per_user = df.groupby('UserID').size().reset_index(name='NumRatings')

fig = px.histogram(
    ratings_per_user,
    x='NumRatings',
    nbins=50,
    title='Ratings Given per User',
    color_discrete_sequence=['#DD8452']
)

fig.update_layout(
    xaxis_title='# Ratings',
    yaxis_title='# Users',
    bargap=0.05
)

fig.show(renderer='colab')

# user demopraphics
age_order = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
age_map = {1: "Under 18", 18: "18-24", 25: "25-34", 35 : "35-44", 45 : "45-49", 50: "50-55", 56: "56+" }

occ_map = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

df['AgeGroup'] = df['Age'].map(age_map)
df['OccupationaLabel'] = df['Occupation'].map(occ_map)

age_counts = df['AgeGroup'].value_counts().reindex(age_order).reset_index()
age_counts.columns = ['AgeGroup', 'Count']

fig = px.bar(
    age_counts,
    x='AgeGroup',
    y='Count',
    text='Count',
    title='Ratings by Age Group',
    color_discrete_sequence=['#55A868']
)

fig.update_traces(
    texttemplate='%{text:,}',
    textposition='outside'
)

fig.show(renderer='colab')

occ_counts = df['OccupationaLabel'].value_counts().sort_values().reset_index()
occ_counts.columns = ['Occupation', 'Count']

fig = px.bar(occ_counts, x='Occupation', y='Count', text='Count',
             title='Rating by Occupation', color_discrete_sequence=['#C44E52'])
fig.update_traces(
    texttemplate='%{text:,}',
    textposition='outside')
fig.show(renderer='colab')

gender_by_user = users['Gender'].value_counts()
print("Unique users by gender:\n", gender_by_user, f'\n\n% male (unique users): {(users['Gender'] == 'M').mean():.1%}')

# Catalog breadkdown: release decade & genre
movies['ReleaseYear'] = movies['Title'].str.extract(r'\((\d{4})\)$').astype(float)
movies['ReleaseDecade'] = (movies['ReleaseYear'] // 10 * 10).astype('Int64')

decade_counts = movies['ReleaseDecade'].value_counts().sort_index().reset_index()
decade_counts.columns = ['ReleaseDecade', 'Count']
decade_counts['ReleaseDecade'] = decade_counts['ReleaseDecade'].astype(str)

fig = px.bar(decade_counts, x='ReleaseDecade', y='Count', text='Count',
             title='Movies in catalog by release decade', color_discrete_sequence=['#8172B2'])
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.show(renderer='colab')

genre_counts = df.assign(Genres=df['Genres'].str.split('|')).explode('Genres')['Genres'].value_counts().sort_values().reset_index()

genre_counts.columns = ['Genre', 'Count']

fig = px.bar(genre_counts, x='Count', y='Genre', text='Count' ,title='Rating volumne by genre (movies have multiple genres)', color_discrete_sequence=['#937860'])
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.update_layout(xaxis_title='Count', yaxis_title='Genre')
fig.show(renderer='colab')

print("Movies by decade (catalog, unique titles):")
print(movies['ReleaseDecade'].value_counts().sort_index())

# Grouping movies by average rating and number of ratings.

movie_group = df.groupby('Title').agg(AvgRating=('Rating', 'mean'), NumRatings=('Rating', 'count')).reset_index()

print("Top 10 movies by NUMBER of Ratings (most-watched):")
movie_group.sort_values('NumRatings', ascending=False).head(10).reset_index(drop=True).round(2)

print("Top 10 movies by AVERAGE rating, with NO minimum-ratings floow-note how few ratings back most of these up: ")
movie_group.sort_values('AvgRating', ascending=False).head(10).reset_index(drop=True).round(2)

MIN_RATINGS = 100
print(f'Top 10 movies by AVERAGE rating, requiring >= {MIN_RATINGS} ratings (far more trustworthy):')
movie_group[movie_group['NumRatings'] >= MIN_RATINGS].sort_values('AvgRating', ascending=False).head(10).reset_index(drop=True).round(2)

fig = px.scatter(movie_group, x='NumRatings', y='AvgRating', hover_name='Title', opacity=0.4, title='Average rating vs Number of ratings, per movie', labels={'NumRatings': '# rating (popularity)', 'AvgRating': 'average rating (quality)'})
fig.add_hline(y=movie_group['AvgRating'].mean(), line_dash='dot', line_color='gray', annotation_text='overall average rating', annotation_position='bottom right')
fig.add_vline(x=movie_group['NumRatings'].mean(), line_dash='dot', line_color='orange', annotation_text='avg number of ratings', annotation_position='bottom right')
fig.update_layout(xaxis_title='# ratings', yaxis_title='average rating')
fig.show(renderer='colab')

# Age x genre cross-analysis
genre_df = df[['AgeGroup','Genres']].copy()
genre_df['Genres'] = genre_df['Genres'].str.split('|')
genre_df = genre_df.explode('Genres').reset_index(drop=True)

top_genres = genre_df['Genres'].value_counts().head(6).index
sub = genre_df[genre_df['Genres'].isin(top_genres)]

age_genre_ct = pd.crosstab(sub['AgeGroup'], sub['Genres'], normalize='index').reindex(age_order)

fig = px.imshow(age_genre_ct, text_auto='.1%', color_continuous_scale='YlGnBu', aspect='auto', labels=dict(color='Share of age group\s rating'),
                title='Genre preference by age group (row-normalized)')
fig.update_layout(xaxis_title='Genre', yaxis_title='Age Group')
fig.show(renderer='colab')
