import pandas as pd

def get_movielens_columns():
    movie_lens_columns = {
        "movie_id": "movie_id",
        "user_id": "user_id",
        "rating": "rating",
        "genre": "genre"
    }
    return movie_lens_columns

def get_train_df(root_dir="../"):
    return pd.read_pickle(root_dir + "train_df.pkl")

def get_val_df(root_dir="../"):
    return pd.read_pickle(root_dir + "val_df.pkl")

def get_test_df(root_dir="../"):
    return pd.read_pickle(root_dir + "test_df.pkl")

def movie_title_from_id(id, train_df):
    return list(train_df["title"].loc[train_df[movie_lens_columns["movie_id"]] == id])[0]

movie_lens_columns = get_movielens_columns()