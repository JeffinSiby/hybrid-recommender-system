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

# def get_items_user_interacted_w(user_id):
#     return list(train_df.loc[train_df[movie_lens_columns["user_id"]]==user_id][movie_lens_columns["movie_id"]])

# Create user profiles from test set - speeds up evaluation when passed in as second arg to ContentBasedRecommender
def create_user_profiles_from_test(content_based_recommender_model, test_df):
    pd.to_pickle(content_based_recommender_model.create_user_profiles(test_df), "./test_user_profiles_for_eval.pkl")

# Create user profiles from validation set - speeds up evaluation when passed in as second arg to ContentBasedRecommender
def create_user_profiles_from_val(content_based_recommender_model, val_df):
    pd.to_pickle(content_based_recommender_model.create_user_profiles(val_df), "./val_user_profiles_for_eval.pkl")

def get_test_user_profs_for_eval_df(root_dir="./"):
    return pd.read_pickle(root_dir + "test_user_profiles_for_eval.pkl")

def get_val_user_profs_for_eval_df(root_dir="./"):
    return pd.read_pickle(root_dir + "val_user_profiles_for_eval.pkl")

# Get model predictions on values in test_df
def get_cbf_predictions(model, test_df):
    actual = []
    preds = []
    test_df_drop = test_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
    n = test_df_drop.shape[0]
    for i, row in test_df_drop.iterrows():
        print(f'{i+1}/{n}')
        # Get preiction for unseen movie in test set
        pred = model.get_rec_strength_for_movie(
                row[movie_lens_columns["user_id"]],
                int(row[movie_lens_columns["movie_id"]])
                )
        # ignore movie_ids that havent been trained on
        if pred != -1:
            preds.append(pred[0][0])
            actual.append(row[movie_lens_columns["rating"]])
            
    return actual, preds


movie_lens_columns = get_movielens_columns()