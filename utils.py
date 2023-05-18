import pandas as pd
from surprise import dump, Reader, SVD, Dataset
import numpy as np

def get_movielens_columns():
    movie_lens_columns = {
        "movie_id": "movie_id",
        "user_id": "user_id",
        "rating": "rating",
        "genre": "genre"
    }
    return movie_lens_columns

def get_train_df(root_dir="./"):
    return pd.read_pickle(root_dir + "train_df.pkl")

def get_val_df(root_dir="./"):
    return pd.read_pickle(root_dir + "val_df.pkl")

def get_test_df(root_dir="./"):
    return pd.read_pickle(root_dir + "test_df.pkl")

### TFIDF and SVD utils have been copied over due to import errors from running rmse_evaluation.py and main.py - e.g. from TFIDF.utils import ... vs utils import ...

#### SVD
def load_model(file_path="./SVD/svd_algo_dump"):
    _, loaded_algo = dump.load(file_path)
    return loaded_algo

def save_model(model, file_path="./SVD/svd_algo_dump"):
    dump.dump(file_path, algo=model)

def movie_title_from_id(id, train_df):
    return list(train_df["title"].loc[train_df[movie_lens_columns["movie_id"]] == id])[0]

def get_ratings_df(train_df):
    '''Get userid, moveiid and ratings columns
    '''
    ratings_df = train_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
    return ratings_df

### TFIDF
# Index from movieID in tags_movies_df
def get_index_from_movieId(id, tags_movies_df):
    inv_map = {v: k for k, v in tags_movies_df[movie_lens_columns["movie_id"]].to_dict().items()}
    return inv_map[id]

# Movie ID from index in tags_movies_df
def get_movie_id_from_index(index, tags_movies_df):
    return tags_movies_df[movie_lens_columns["movie_id"]].to_dict()[index]

def tfidf_index_to_tags_movies_df_index(i, tags_movies_df):
    return tags_movies_df.index[i]

def tags_movies_df_index_to_tfidf_index(i, tags_movies_df):
    return list(tags_movies_df.index).index(i)

# Get movie title from ID
def movie_title_from_id(id, tags_movies_df):
    return list(tags_movies_df["title"].loc[tags_movies_df[movie_lens_columns["movie_id"]] == id])[0]


movie_lens_columns = get_movielens_columns()