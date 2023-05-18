from Popularity.popularity_rs import PopularityRecommender
from SVD.cf_rs import CollaborativeFilteringRecommender
from SVD.utils import load_model
from TFIDF.cbf_rs import ContentBasedRecommender
from hybrid_rs import HybridRS
from utils import get_movielens_columns, get_train_df
import pandas as pd
import textwrap

def print_explanation_of_np_results(top_n):
    print("\nEXPLANATION OF RESULTS")
    print('-------------------------------------------------------------------------------------')
    print(f'A list of the {top_n} most popular movies amongst users are displayed below with the following columns:')
    print(f'''1) movie_id -> Unique movie ID\n2) title -> Movie title\n3) rating -> Predicted rating out of 5 based on the popularity and average rating of the movie\n4) percentage -> The percentage we think you will enjoy the movie based on the rating\n''')

def gen_recs_df_to_display(model, top_n):
    recs_df = model.recommend_movies(user_id, top_n)
    recs_df["percentage"] = recs_df["rating"].apply(lambda x: f'{((x/5)*100):.2f}%')
    recs_df.index +=1
    return recs_df

if __name__ == "__main__":

    print("Please wait for the system to load...")
    

    # Initialise on startup so user has less waiting time
    train_df = get_train_df()
    movie_lens_columns = get_movielens_columns()
    all_user_ids = train_df[movie_lens_columns["user_id"]].unique()

    svd_model = load_model("./SVD/svd_algo_80_dump")

    # Initialise hybrid model
    cf_model = CollaborativeFilteringRecommender(svd_model, train_df)
    cbf_model = ContentBasedRecommender(train_df)
    hybrid_model = HybridRS(cbf_model, cf_model, 0.1, train_df)
    pop_rs = PopularityRecommender(train_df)

    TOP_N = 5
    MIN_RATINGS = 10
    # Display all column content
    pd.set_option('display.max_colwidth', None)

    print("\n================================= Movie Recommender =================================")
    logged_in = False
    while logged_in == False:
        print("\nPlease login before using the system:")
        user_id = -1
        password = ""
        first = True

        # Check if user name and password are correct
        while (password != "pass1") or (user_id not in all_user_ids):
            # Display message if the user has had an attempt
            if not first:
                print("User ID or password is incorrect. Please try again.\n")
            first = False
            user_id = input("Enter your user ID: ")
            if user_id.isdigit():
                user_id = int(user_id)
            else:
                user_id = -1
            password = input("Enter your password: ")
        # if user_id in all_user_ids:
        logged_in = True
        print('-------------------------------------------------------------------------------------')
        print(f'\nWELCOME BACK USER {user_id},')
        # print(train_df.loc[movie_lens_columns["user_id"] == 1])
        movies_rated = len(train_df.loc[train_df[movie_lens_columns["user_id"]] == user_id][movie_lens_columns["movie_id"]].unique())
        print(f'You have rated {movies_rated} movies. The more movies you rate, the more accurate our recommendations get')

        non_personalised = movies_rated < MIN_RATINGS
        # Users who dont get personalised recommendations
        if non_personalised:
            print("\nPLEASE NOTE")
            print('-------------------------------------------------------------------------------------')
            print(f'''The recommendations you're given is based on what's popular among all users.\nPlease rate {MIN_RATINGS-movies_rated} more movies to get personalised recommendations.''')
            
            print_explanation_of_np_results(TOP_N)

            pop_df = gen_recs_df_to_display(pop_rs, TOP_N)
            print(f'------------------------------------------RECOMMENDATIONS------------------------------------------')
            print(pop_df)
        # Users who get personalised recommendations
        else:
            print("\nEXPLANATION OF RESULTS")
            print('-------------------------------------------------------------------------------------')
            print(F'A list of top {TOP_N} movie recommendations are displayed below with the following columns:')
            print(f'''1) movie_id -> Unique movie ID\n2) title -> Movie title\n3) rating -> Predicted rating out of 5 based on:\n    - Movies with similar tags and\n    - Ratings of other users\n4) percentage -> The percentage we think you will enjoy the movie based on the rating\n''')
            print("\nIt may take a couple of seconds to load movie recommendations specially tailored for you...\n")
            hybrid_df = gen_recs_df_to_display(hybrid_model, TOP_N)

            print(f'------------------------------------------RECOMMENDATIONS------------------------------------------')
            print(hybrid_df)
        
        # Top N already produced once so increase count
        top_n_count = 1

        while logged_in:
            personalised_g_text = f'[G] - Generate {TOP_N} MORE personalised movie recommendations'
            # If the user requested for option 'P', reset the personalised recommendation count
            if top_n_count == 0:
                personalised_g_text = f'[G] - Generate {TOP_N} personalised movie recommendations'

            personalised_options = {
                "G" : personalised_g_text,
                "P" : "[P] - Show me what's popular",
                "F" : '[F] - Find out more about how we use your data',
                "L" : '[L] - Logout',
                "X" : '[X] - Exit'
            }

            non_personalised_options = {
                "G" : f'[G] - Generate {TOP_N} more popular movie recommendations',
                "F" : '[F] - Find out more about how we use your data',
                "L" : '[L] - Logout',
                "X" : '[X] - Exit'
            }

            options = non_personalised_options if non_personalised else personalised_options

            option_keys = list(options.keys())

            option = -1
            print("\nOPTIONS")
            while option not in option_keys:
                for key in option_keys:
                    print(f'{options[key] }')
                option = input('Please choose an option from the list above to get started: ')
                option = option.upper()
                print('')
            
            if option == "G" or option == "P":
                # Generate top_n more recommendations (can be for personalised user or non personalised)
                if option == "G":
                    top_n_count += 1
                    if non_personalised:
                        print(f'------------------------------------------RECOMMENDATIONS------------------------------------------')
                        pop_df = gen_recs_df_to_display(pop_rs, TOP_N*top_n_count)
                        print(pop_df.tail(TOP_N))
                    else:
                        hybrid_df = gen_recs_df_to_display(hybrid_model, TOP_N*top_n_count)
                        print(f'------------------------------------------RECOMMENDATIONS------------------------------------------')
                        print(hybrid_df.tail(TOP_N))
                # Show top_n popular movies - only available for personalised users
                elif option == "P":
                    # Reset the personalised count since popularity model recommendation was just given
                    top_n_count = 0
                    print_explanation_of_np_results(TOP_N)
                    pop_df = gen_recs_df_to_display(pop_rs, TOP_N)
                    print(f'------------------------------------------RECOMMENDATIONS------------------------------------------')
                    print(pop_df)
            
            # Logout
            elif option == "L":
                logged_in = False

            elif option == "F":
                print("\nWhat data do we collect?")
                print("----------------------------")
                print(textwrap.fill('Your privacy is very important and we recognise that. The only information we store are your userIDs and passwords, and a record of your ratings for a movie',100))

                print("\nWhy do we store it?")
                print("----------------------------")
                print(textwrap.fill('We store your userID and password for login purposes. Your unique userID along with a history of your ratings is used to provide you personalised recommendations based on what you and other users have interacted with.', 100))

            # Exit
            elif option == "X":
                break
        