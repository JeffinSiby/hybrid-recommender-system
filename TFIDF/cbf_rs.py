import pandas as pd
from utils import get_movielens_columns, get_index_from_movieId, get_movie_id_from_index, movie_title_from_id, tags_movies_df_index_to_tfidf_index, tfidf_index_to_tags_movies_df_index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    '''create_all_user_profiles used for evaluation to speed up querying multiple users
    train_df used to create TFIDF matrix
    '''
    def __init__(self, train_df, all_user_profiles=None):
        self.train_df = train_df
        self.tags_movies_df = self.gen_tags_movies_df()
        self.tfidf_df = self.gen_tfidf()
        self.all_movie_ids = self.tags_movies_df[movie_lens_columns["movie_id"]].unique()
        self.user_profiles = all_user_profiles
    
    # DF containing unique movieids and corresonding tags. 'document' column also generated. Used for creating tfidf matrix
    def gen_tags_movies_df(self):
        tags_movies_df = self.train_df.drop_duplicates(subset=movie_lens_columns["movie_id"])
        tags_movies_df = tags_movies_df.drop(movie_lens_columns["user_id"], axis=1)
        tags_movies_df['document'] = tags_movies_df[['tag', movie_lens_columns["genre"]]].apply(lambda x: ' '.join(x), axis=1)
        tags_movies_df = tags_movies_df.drop([movie_lens_columns["genre"],"tag"], axis=1)
        return tags_movies_df
    
    def gen_tfidf(self):
        tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
        tfidf_matrix = tfidfvectorizer.fit_transform(self.tags_movies_df['document'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray())
        return tfidf_df

    def get_similar_movies_to_user_prof(self, user_id, top_n=10):
        #Computes the cosine similarity between the user profile and all movie profiles and returns array of similarity of each movie
        cosine_similarities = cosine_similarity(np.array((
            self.create_user_profile(user_id) if self.user_profiles is None else self.user_profiles[user_id]
            ), ndmin=2), self.tfidf_df)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-top_n:]
        #Sort the similar items by similarity. movieId and cosine similarity returned as a tuple in list
        similar_items = sorted([
            (get_movie_id_from_index(
                tfidf_index_to_tags_movies_df_index(i, self.tags_movies_df),
                self.tags_movies_df
                ),
            movie_title_from_id(
                get_movie_id_from_index(
                    tfidf_index_to_tags_movies_df_index(i, self.tags_movies_df),
                    self.tags_movies_df
                    ),
                    self.tags_movies_df),
            # Multiple similarity by 5 to get rating
            cosine_similarities[0,i] * 5) for i in similar_indices], key=lambda x: -x[2])
        return similar_items

    def recommend_movies(self, user_id, top_n=10):
        # Ignore items user already interacted with
        items_to_ignore = list(self.train_df.loc[self.train_df[movie_lens_columns["user_id"]]==user_id][movie_lens_columns["movie_id"]])
        similar_items = self.get_similar_movies_to_user_prof(user_id, top_n*3)
        # Inspired from Kaggle-SVD.ipynb shown in lectures
        #Ignores movie IDs the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=[movie_lens_columns["movie_id"], 'title', 'rating']).head(top_n)

        return recommendations_df
    
    # Find cosine similarity of given movie to user id
    def get_rec_strength_for_movie(self, user_id, movie_id):
        if movie_id in self.all_movie_ids:
            tags_movies_df_index_of_movie = (self.tags_movies_df.index[self.tags_movies_df[movie_lens_columns["movie_id"]] == movie_id])[0]
            tfidf_row = self.tfidf_df.iloc[tags_movies_df_index_to_tfidf_index(
                tags_movies_df_index_of_movie,
                self.tags_movies_df
                )]
            # Multiply by 5 to get rating from cosine similarity
            return cosine_similarity(np.array(
                self.user_profiles[user_id] if self.user_profiles!=None else self.create_user_profile(user_id)
                , ndmin=2), np.array(tfidf_row, ndmin=2)) * 5
        else:
            # movie ID doesnt exist
            return -1
    
    # Get TFIDF vector of a movie
    def get_movie_profile(self, movie_id):
        if movie_id in self.tags_movies_df[movie_lens_columns["movie_id"]].unique():
            return self.tfidf_df.iloc[tags_movies_df_index_to_tfidf_index(
                get_index_from_movieId(movie_id, self.tags_movies_df),
                self.tags_movies_df
                )]
        else:
            print(f"movie id {movie_id} not found in tags_movies_df")
            return pd.Series(np.zeros((self.tfidf_df.shape[1])))

    def create_user_profile(self, user_id):
        user_profile = pd.Series(np.zeros((self.tfidf_df.shape[1])))
        ratings_df = self.train_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
        # get movies user interacted with to create user profile
        for movie_id, rating in zip(
            ratings_df.loc[ratings_df[movie_lens_columns["user_id"]] == user_id][movie_lens_columns["movie_id"]],
            ratings_df.loc[ratings_df[movie_lens_columns["user_id"]] == user_id]['rating']):
            if float(rating) <= 3:
                user_profile -= self.get_movie_profile(movie_id)
            else:
                user_profile += self.get_movie_profile(movie_id)
        # Normalise
        user_profile_norm = (user_profile - user_profile.mean()) / (user_profile.max() - user_profile.min())
        return user_profile_norm

    # Create all user profiles
    # from_df is the train df by default but can be changed to e.g. test df to only generate user profiles in test set for evaluation
    def create_user_profiles(self, from_df=None):
        if from_df is None:
            from_df = self.train_df
        user_profiles = {}
        # Create user profile for each user
        ratings_df = from_df[[movie_lens_columns["user_id"], movie_lens_columns["movie_id"], movie_lens_columns["rating"]]]
        unique_users_df = ratings_df[movie_lens_columns["user_id"]].unique()
        for i, user_id in enumerate(unique_users_df):
            print(f'Completed for user {i+1}/{unique_users_df.shape[0]}')
            user_profiles[user_id] = self.create_user_profile(user_id)
        return user_profiles

    def update_train_df(self, new_train_df):
        self.train_df = new_train_df

movie_lens_columns = get_movielens_columns()