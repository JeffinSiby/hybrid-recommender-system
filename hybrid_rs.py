from Popularity.popularity_rs import PopularityRecommender
from SVD.cf_rs import CollaborativeFilteringRecommender
from SVD.utils import load_model
from TFIDF.cbf_rs import ContentBasedRecommender
from utils import get_movielens_columns, get_train_df
import pandas as pd

class HybridRS:
    '''train_df used to check if user has less than m ratings - if so use popularity model
    '''
    def __init__(self, cbf_model, cf_model, cbf_model_weight, train_df):
        self.cbf_model = cbf_model
        self.cf_model = cf_model
        self.cbf_model_weight = min(1, cbf_model_weight)
        self.cf_model_weight = 1 - cbf_model_weight
        self.train_df = train_df

    def recommend_movies(self, user_id, top_n=10):
        cbf_rs_recommendations = self.cbf_model.recommend_movies(user_id, 200).rename(columns={'rating': 'ratingCBF'})
        cf_rs_recommendations = self.cf_model.recommend_movies(user_id, 200).rename(columns={'rating': 'ratingCF'})
        #Combining the results by contentId
        recs_df = cbf_rs_recommendations.merge(cf_rs_recommendations,
                                how = 'outer', 
                                left_on = [movie_lens_columns["movie_id"], "title"], 
                                right_on = [movie_lens_columns["movie_id"], "title"]).fillna(0.0)
        
        recs_df['rating'] = (recs_df['ratingCBF'] * self.cbf_model_weight) \
                                    + (recs_df['ratingCF'] * self.cf_model_weight)
        recs_df = recs_df[[movie_lens_columns["movie_id"], "title", "rating"]]
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('rating', ascending=False).head(top_n)
        return recommendations_df.reset_index(drop=True)
    
    def get_rec_strength_for_movie(self, user_id, movie_id):
        cbf_rec_strength = self.cbf_model.get_rec_strength_for_movie(user_id, movie_id)
        cf_rec_strength = self.cf_model.get_rec_strength_for_movie(user_id, movie_id)
        if cbf_rec_strength == -1 or cf_rec_strength == -1:
            return -1
        return ((cbf_rec_strength*self.cbf_model_weight) + (cf_rec_strength*self.cf_model_weight))[0][0]


movie_lens_columns = get_movielens_columns()

# if __name__ == "__main__":
#     train_df = get_train_df()
#     svd_model = load_model("./SVD/svd_algo_80_dump")
#     cf_model = CollaborativeFilteringRecommender(svd_model, train_df)
#     cbf_model = ContentBasedRecommender(train_df)

#     hybrid_model = HybridRS(cbf_model, cf_model, 0.05, train_df)
#     print(hybrid_model.recommend_movies(2, 10))
#     print(hybrid_model.get_rec_strength_for_movie(2, 1148))
#     print(hybrid_model.get_rec_strength_for_movie(2, 745))
