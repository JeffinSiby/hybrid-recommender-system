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

def get_train_df(root_dir="../"):
    return pd.read_pickle(root_dir + "train_df.pkl")

def get_val_df(root_dir="../"):
    return pd.read_pickle(root_dir + "val_df.pkl")

def get_test_df(root_dir="../"):
    return pd.read_pickle(root_dir + "test_df.pkl")

def get_clean_dfs(root_dir="../"):
    movies_df_w_dropped_movieID = pd.read_pickle(root_dir + "cleaned_dropped_movies.pkl")
    tags_df_w_dropped_movieID = pd.read_pickle(root_dir + "cleaned_dropped_tags.pkl")
    ratings_df_w_dropped_movieID = pd.read_pickle(root_dir + "cleaned_dropped_ratings.pkl")
    grouped_tag_df = tags_df_w_dropped_movieID.groupby(movie_lens_columns["movie_id"])["tag"].apply(list).map(lambda x: " ".join(x)).reset_index(name='tag')

    ratings_movies_df = pd.merge(ratings_df_w_dropped_movieID,  movies_df_w_dropped_movieID, on=movie_lens_columns["movie_id"])
    test_train = pd.merge(ratings_movies_df,  grouped_tag_df, on=movie_lens_columns["movie_id"])
    return test_train
    
def get_ratings_df(train_df):
    '''Get userid, moveiid and ratings columns'''
    ratings_df = train_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
    return ratings_df

def load_model(file_path="./svd_algo_dump"):
    _, loaded_algo = dump.load(file_path)
    return loaded_algo

def save_model(model, file_path="./svd_algo_dump"):
    dump.dump(file_path, algo=model)

def movie_title_from_id(id, train_df):
    return list(train_df["title"].loc[train_df[movie_lens_columns["movie_id"]] == id])[0]

def get_ratings_df(train_df):
    '''Get userid, moveiid and ratings columns
    '''
    ratings_df = train_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
    return ratings_df

def train_svd(train_df, n_factors=80):
    # Keep predictions in scale 1-5
    reader = Reader(rating_scale=(1,5))
    ratings_df = get_ratings_df(train_df)
    data = Dataset.load_from_df(ratings_df, reader)
    data = data.build_full_trainset()
    svd_algo = SVD(n_factors=n_factors, n_epochs=20, random_state=4)
    svd_algo.fit(data)
    return svd_algo

# Generate models with different latent factors for eval
def gen_models_for_testing(train_df):
    for latent_factors in [20,40,60,80,100]:
        new_model = train_svd(train_df, latent_factors)
        save_model(new_model, file_path=f'./svd_algo_{latent_factors}_dump')

def get_predictions(cf_model, test_df):
    actual = []
    preds = []
    for _, row in test_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]].iterrows():
        # Get preiction for unseen movie in test set
        pred = cf_model.get_rec_strength_for_movie(
                row[movie_lens_columns["user_id"]],
                int(row[movie_lens_columns["movie_id"]])
                )
        # ignore movie_ids that havent been trained on
        if pred != -1:
            preds.append(pred)
            actual.append(row[movie_lens_columns["rating"]])
            
    return actual, preds

# Genrate predictions dataframe for each mode in 'models' array. Use validation dataset for this.
# model_names = array of model names
def gen_preds_for_models(models, val_df, model_names):
    results = []
    for model in models:
        actual, preds = get_predictions(model, val_df)
        results.append(preds)
    results.append(actual)
    results = np.array(results)
    columns = model_names + ["actual"]
    predictions_df = pd.DataFrame(results.transpose(), columns=columns)
    pd.to_pickle(predictions_df, "lf_val_predictions.pkl")

movie_lens_columns = get_movielens_columns()