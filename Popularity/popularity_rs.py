from utils import get_movielens_columns

class PopularityRecommender:
    def __init__(self, train_df):
        self.train_df = train_df
        self.popularity_model = self.gen_popularity_df(self.train_df)

    # inspired by https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/notebook
    def weighted_rating(self, x, total_avg):
        C = total_avg
        m = 1 # minimum votes required to be listed
        v = x["count"]
        R = x["mean"]
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Traindf can be changed to test for testing to create 
    # generate dataframe containing most popular movies ranked by weighted rating
    def gen_popularity_df(self, train_df):
        popularity_model = train_df[[movie_lens_columns["movie_id"], "title", movie_lens_columns["rating"]]].groupby(movie_lens_columns["movie_id"]).agg({movie_lens_columns["rating"]: ['count','mean'], "title":"first"}).reset_index()#.mean().reset_index()
        total_mean = popularity_model["rating"]["mean"].mean()
        # dirty method of keeping title - TODO: performance improvement with alternative methods
        popularity_model["single_title"] = popularity_model["title"]["first"]
        popularity_model["weighted_rating"] = popularity_model["rating"].apply(lambda x: self.weighted_rating(x, total_mean), axis=1)
        popularity_model = popularity_model.sort_values('weighted_rating', ascending=False)
        popularity_model = popularity_model[[movie_lens_columns["movie_id"],"single_title", "weighted_rating"]]
        popularity_model["rating"] = popularity_model["weighted_rating"]
        popularity_model["title"] = popularity_model["single_title"]
        return popularity_model[[movie_lens_columns["movie_id"],"title", "rating"]]

    def recommend_movies(self, user_id, top_n=10):
        #movies user has interacted with
        movies_to_ignore = self.train_df.loc[self.train_df[movie_lens_columns["user_id"]]==user_id][movie_lens_columns["movie_id"]].tolist()
        filtered_popularity_df = self.popularity_model[~self.popularity_model[movie_lens_columns["movie_id"]].isin(movies_to_ignore)]
        top_n_df = filtered_popularity_df.head(top_n).reset_index()
        return top_n_df[[movie_lens_columns["movie_id"], "title", "rating"]]

    # get rating for a movie id based on popularity
    def get_rec_strength_for_movie(self, movie_id):
        rating_series = self.popularity_model.loc[self.popularity_model[movie_lens_columns["movie_id"]] == movie_id]["rating"]
        # return -1 if movie not found
        if len(rating_series) == 0:
            return -1
        else:
            return rating_series.values[0]

# train_df = get_train_df("../")
movie_lens_columns = get_movielens_columns()
# popularity_model = PopularityRecommender(train_df)