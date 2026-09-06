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

avg_rating_by_genre =(
    df.assign(Genres=df['Genres'].str.split('|'))
    .explode("Genres")
    .groupby('Genres')['Rating']
    .mean()
    .sort_values(ascending=False)
)
print("Average rating by genre:")
print(avg_rating_by_genre.round(3))

# Data Preprocessing & Feature Engineering 

# Cleaning and formatting 
# Timestamp -> clander featues
df['RatingDatetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df['RatingYear'] = df['RatingDatetime'].dt.year
df['RatingMonth'] = df['RatingDatetime'].dt.month
df['RatingDay'] = df['RatingDatetime'].dt.day
df['RatingDayOfWeek'] = df['RatingDatetime'].dt.day_name()
df['isWeekend'] = df['RatingDatetime'].dt.dayofweek.isin([5,6])

cal = USFederalHolidayCalendar()
us_holidays = cal.holidays(start = df['RatingDatetime'].min(), end = df['RatingDatetime'].max())
df['isHoliday'] = df['RatingDatetime'].dt.normalize().isin(us_holidays)

print(f'Rating span: {df['RatingDatetime'].min().date()} to {df['RatingDatetime'].max().date()}')
print(f'Weekend ratings: {df['isWeekend'].mean():.1%} | Holiday ratings: {df['isHoliday'].mean():.1%}')

# Release Year / Decade, extrated from the movie title(titles follow "Title (YYYY)")
df['ReleaseYear'] = df['Title'].str.extract(r'\((\d{4})\)$').astype(float)
df['ReleaseDecade'] = (df['ReleaseYear'] // 10 * 10).astype('Int64')


# Approximate US region from the leading ZIP digit (standard USPS leading-digit grouping). This is a coarse "State_US"-style geography feature — not exact state-level, since that would require an external ZIP->state lookup table, but useful for regional exploration.
zip_region_map = {'0': 'Northeast', '1': 'Northeast', '2': 'Mid-Atlantic South', '3': 'Southeast',
                   '4': 'Midwest (East)', '5': 'Midwest (West)', '6': 'Central', '7': 'South Central',
                   '8': 'Mountain', '9': 'Pacific'}
df['State_US_Region'] = df['Zip-code'].astype(str).str.strip().str[0].map(zip_region_map)

df[['Title', 'ReleaseYear', 'ReleaseDecade', 'Zip-code', 'State_US_Region', 'RatingDayOfWeek', 'isWeekend', 'isHoliday']].head()

# Derived aggregate features
movie_stats = df.groupby('MovieID').agg(AvgRatingMoive=('Rating','mean'), NumRatingMovie=('Rating','count')).reset_index()

user_stats = df.groupby('UserID').agg(AvgRatingUser=('Rating','mean'),
NumRatingUser=('Rating','count')).reset_index()

df = df.merge(movie_stats, on='MovieID').merge(user_stats, on='UserID')

df[['Title', 'ReleaseYear', 'ReleaseDecade', 'Zip-code', 'State_US_Region', 'RatingDayOfWeek', 'isWeekend', 'isHoliday']].head()

# Encoding Categorical data
# Genres: Multi-label one-hot endocing (a movie can belong to several genres at once)
genre_dummies = df['Genres'].str.get_dummies(sep='|')
print('Genre one-hot matrix:', genre_dummies.shape)
genre_dummies.head(3)

pivot = df.pivot_table(index='UserID', columns='MovieID', values='Rating')
pivot_filled = pivot.fillna(0)
print("Use-item pivot table (user x movies): ",pivot_filled.shape)
print(f'Confirmed sparsityL {(pivot.isna().sum().sum() / pivot.size):.2%} missing')

# Model 1 Item-item Collaborative Filtering (Pearson Correlation)
pivot = df.pivot_table(index='UserID', columns='Title', values='Rating')
pivot_filled = pivot.fillna(0)

movie_rating_counts = df.groupby('Title').size()

def recommend_pearson(movie_name, n=5, min_ratings=50):
  """Top-n movies most correlated with 'movie-name' by Pearson Correlation of rating vectors."""
  if movie_name not in pivot_filled.columns:
    close = [c for c in pivot_filled.columns if movie_name.lower() in c.lower()]
    raise ValueError(f"'{movie_name}' not found. Close matches: {close[:5]}")
  target = pivot_filled[movie_name]
  corrs = pivot_filled.corrwith(target)
  out = pd.DataFrame({'Title': corrs.index, 'PearsonCorrelation': corrs.values})
  out = out.merge(movie_rating_counts.rename('NumRatings'), left_on='Title', right_index=True)
  out = out[(out['Title'] != movie_name) & (out['NumRatings'] >= min_ratings)]
  return out.sort_values('PearsonCorrelation', ascending=False).head(n).reset_index(drop=True)

pearson_liarliar = recommend_pearson('Liar Liar (1997)', n=5)
pearson_liarliar

# A couple more examples to sanity check the recommdender across genres.
for title in ['Star Wars: Episode IV - A New Hope (1977)', 'Toy Story (1995)']:
  print(f"\nTop 5 similar to: {title}")
  print(recommend_pearson(title, n=5).to_string(index=False))

# Model 2 Collaborative Filtering with Cosine Similarity
item_sim = cosine_similarity(pivot_filled.T.values)
item_sim_df = pd.DataFrame(item_sim, index=pivot_filled.columns, columns=pivot_filled.columns)
print("Item-item similarity matrix: ", item_sim_df.shape)
print("Example: item-item similarity for thei first 5 movies")
item_sim_df.iloc[:5,:5].round(3)

user_sim = cosine_similarity(pivot_filled.values)
user_sim_df = pd.DataFrame(user_sim, index=pivot_filled.index, columns=pivot_filled.index)
print("User-user similarity matrix: ", user_sim_df.shape)
print("Example: User-user similarity for the first 5 users")
user_sim_df.iloc[:5,:5].round(3)

def recommend_cosine(movie_name, n=5, min_ratings=50):
  """Top-n movies most similar to 'movie-name' by Cosine Similarity of rating vectors."""
  sims = item_sim_df[movie_name].drop(index=movie_name)
  out = pd.DataFrame({'Title': sims.index, 'CosineSimilarity': sims.values})
  out = out.merge(movie_rating_counts.rename('NumRatings'), left_on='Title', right_index=True)
  out = out[out['NumRatings'] >= min_ratings]
  return out.sort_values('CosineSimilarity', ascending=False).head(n).reset_index(drop=True)

cosine_liarliar = recommend_cosine('Liar Liar (1997)', n=5)
cosine_liarliar

# Nearest Neighbors on CSR (sparse) matrix.
movie_to_idx = {title: i for i, title in enumerate(pivot_filled.columns)}
idx_to_movie = {i: title for title, i in movie_to_idx.items()}

item_user_csr = csr_matrix(pivot_filled.T.values)
print(f"CSR matrix: {item_user_csr.shape}, {item_user_csr.nnz:,} stored (non-zero) entries "
      f"out of {item_user_csr.shape[0]*item_user_csr.shape[1]:,} possible "
      f"({item_user_csr.nnz / (item_user_csr.shape[0]*item_user_csr.shape[1]):.2%} dense)")

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(item_user_csr)

def recommend_knn(movie_name, n=5):
  idx = movie_to_idx[movie_name]
  distances, indices = knn_model.kneighbors(item_user_csr[idx], n_neighbors= n+1)
  recs = [(idx_to_movie[i], 1 - d) for d, i in zip(distances.flatten(), indices.flatten()) if i != idx]
  return pd.DataFrame(recs, columns=['Title', 'CosineSimilarity']).head(n)

knn_liarliar = recommend_knn('Liar Liar (1997)', n=5)
knn_liarliar

# Worked examples: sparse-matrix representation
example_dense = np.array([[1, 0], [3, 7]])
example_csr = csr_matrix(example_dense)
print("Dense matrix:\n", example_dense)
print("\nCSR components:")
print("  data (non-zero values, row-major order):", example_csr.data)
print("  indices (column index of each value):   ", example_csr.indices)
print("  indptr (start offset of each row):       ", example_csr.indptr)
print("\n", example_csr)

# Model 3 - Matrix Factorization (SVD)
ratings_for_mf = df[['UserID', 'MovieID', 'Rating']].drop_duplicates(subset=['UserID', 'MovieID'])
reader = Reader(rating_scale=(1,5))
mf_data = Dataset.load_from_df(ratings_for_mf, reader)

mf_trainset, mf_testset = surprise_train_test_split(mf_data, test_size=0.2, random_state=RANDOM_STATE)

svd = SVD(n_factors=4, random_state=RANDOM_STATE)
svd.fit(mf_trainset)

predictions = svd.test(mf_testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

y_true = np.array([p.r_ui for p in predictions])
y_pred = np.array([p.est for p in predictions])
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"Matrix Factorization (d=4) — held-out test set of {len(predictions):,} ratings")
print(f"  RMSE: {rmse:.4f}  (average star-rating error, penalizing large misses more)")
print(f"  MAE:  {mae:.4f}  (average absolute star-rating error)")
print(f"  MAPE: {mape:.2f}%  (average error as a % of the true rating)")

## Embedding-Based Similarity & Visualization
### Item-item similarity from d=4 embeddings
item_factors = svd.qi  # shape: (n_items_in_trainset, 4)

movieid_to_title = df.drop_duplicates('MovieID').set_index('MovieID')['Title'].to_dict()

inner_to_title = {
    mf_trainset.to_inner_iid(raw_iid): movieid_to_title.get(raw_iid, f"MovieID {raw_iid}")
    for raw_iid in mf_trainset._raw2inner_id_items
}
titles_ordered = [inner_to_title[i] for i in range(item_factors.shape[0])]

item_embedding_sim = pd.DataFrame(cosine_similarity(item_factors), index=titles_ordered, columns=titles_ordered)

def recommend_embedding(movie_name, n=5):
    sims = item_embedding_sim[movie_name]
    if isinstance(sims, pd.DataFrame):  # guard in the rare case of duplicate titles
        sims = sims.iloc[:, 0]
    return sims.drop(index=movie_name).sort_values(ascending=False).head(n)

embedding_liarliar = recommend_embedding('Liar Liar (1997)', n=5)
embedding_liarliar
