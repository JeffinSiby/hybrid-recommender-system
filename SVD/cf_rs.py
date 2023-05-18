import pandas as pd
from utils import get_movielens_columns, movie_title_from_id

class CollaborativeFilteringRecommender:
    def __init__(self, model, train_df):
        self.model = model
        self.train_df = train_df
        self.all_movies = train_df[movie_lens_columns["movie_id"]].unique()
    
    # get top_n movie_ids user has not interacted w/ in the form (movie_id, rating)[]
    def get_top_for_user(self, user_id, top_n=10):
        uninteracted_ratings = []
        # For each movie_id user has not interacted w/
        for uninteracted_movieid in set(self.all_movies) - set(self.train_df.loc[self.train_df[movie_lens_columns["user_id"]] == user_id][movie_lens_columns["movie_id"]].unique()):
            self.model.predict(user_id,uninteracted_movieid)[3]
            uninteracted_ratings.append((
                uninteracted_movieid,
                self.model.predict(user_id,uninteracted_movieid)[3]
                ))

        return sorted(uninteracted_ratings, key=lambda x: x[1], reverse=True)[:top_n]

    # Generate recommendations df
    def recommend_movies(self, user_id, top_n=10):
        top_n_movieids = self.get_top_for_user(user_id, top_n)
        return pd.DataFrame([(
                movie_id,
                movie_title_from_id(movie_id, self.train_df),
                rating
                ) for (movie_id,rating) in top_n_movieids], columns=['movie_id', 'title', 'rating'])

    # Get recommendation strength for given movie for given user - used for rating evaluation
    def get_rec_strength_for_movie(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id)[3]
    
movie_lens_columns = get_movielens_columns()