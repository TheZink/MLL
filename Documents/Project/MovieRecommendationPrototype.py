import time
import pandas as pd
from surprise import Reader, Dataset, KNNWithZScore
from surprise import KNNBasic

# Load the datasets
movies = pd.read_csv('datasets/movies-small/ml-latest-small/movies.csv', sep=',')
ratings = pd.read_csv('datasets/movies-small/ml-latest-small/ratings.csv', sep=',')

# Extract the year from the title column using a regular expression
year_str = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
# Remove the year from the title column
movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
# Create a new column for the year and convert it to numeric
movies['year'] = pd.to_numeric(year_str, errors='coerce').astype('Int64')

# Create a merged dataset of ratings and movies
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Check for null values
null_values = movie_ratings.isnull().sum()
# Remove null rows if there are any
if null_values.any():
    movie_ratings = movie_ratings.dropna()

# Check for duplicates
duplicates = movie_ratings.duplicated().sum()
if duplicates > 0:
    movie_ratings = movie_ratings.drop_duplicates()

# Convert timestamp to datetime
movie_ratings['timestamp'] = pd.to_datetime(movie_ratings['timestamp'], unit='s')

# adding new ratings to the ratings dataset from new user
new_user_id = movie_ratings['userId'].max() + 1
new_ratings = pd.DataFrame({
    'userId': [new_user_id] * 16,
    'movieId': [
        188301, # Ant-Man and the Wasp (2018)
        187595, # Solo: A Star Wars Story (2018)
        185029, # A Quiet Place (2018)
        184471, # Tomb Raider (2018)
        176371, # Blade Runner 2049 (2017)
        175303, # It (2017)
        168250, # Get Out (2017)
        168248, # John Wick: Chapter Two (2017)
        167370, # Assassin's Creed (2016)
        166528, # Rogue One: A Star Wars Story (2016)
        166024, # Whiplash (2013)
        163645, # Hacksaw Ridge (2016)
        161131, # War Dogs (2016)
        139385, # The Revenant (2015)
        122916, # Thor: Ragnarok (2017)
        122904, # Deadpool (2016)
    ],
    'rating': [
        2.75, # Ant-Man and the Wasp (2018)
        4.0, # Solo: A Star Wars Story (2018)
        4.8, # A Quiet Place (2018)
        3.75, # Tomb Raider (2018)
        5.0, # Blade Runner 2049 (2017)
        3.90, # It (2017)
        4.5, # Get Out (2017)
        4.75, # John Wick: Chapter Two (2017)
        1.75, # Assassin's Creed (2016)
        3.75, # Rogue One: A Star Wars Story (2016)
        4.95, # Whiplash (2013)
        4.85, # Hacksaw Ridge (2016)
        4.75, # War Dogs (2016)
        4.25, # The Revenant (2015)
        3.80, # Thor: Ragnarok (2017)
        4.0, # Deadpool (2016)
    ],
    'timestamp': [int(time.time())] * 16
})
# Add the titles, genres and year for the new ratings
new_ratings = pd.merge(new_ratings, movies[['movieId', 'title', 'genres', 'year']], on='movieId', how='left')
movie_ratings = pd.concat([movie_ratings, new_ratings], ignore_index=True)

# Build the recommendation model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)

sim_options = {
    'name': 'cosine',  # similarity measure
    'user_based': True,  # user-based CF
}

final_trainset = data.build_full_trainset()
final_model = KNNWithZScore(sim_options=sim_options, k=5)
final_model.fit(final_trainset)

print("\n" + "=" * 60)
print("MOVIE RECOMMENDATION SYSTEM - Model trained successfully!")
print("=" * 60)


def get_top_n_recommendations(user_id, n=10):
    # Get all movies the user hasn't rated yet
    rated_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].unique()
    all_movies = movies['movieId'].unique()
    unrated_movies = [m for m in all_movies if m not in rated_movies]

    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        try:
            pred = final_model.predict(user_id, movie_id)
            movie_info = movies[movies['movieId'] == movie_id].iloc[0]
            predictions.append((movie_id, pred.est, movie_info['title'], movie_info['genres']))
        except:
            continue

    # Sort by predicted rating and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def search_movie(title_query):
    matches = movies[movies['title'].str.contains(title_query, case=False, na=False)]
    return matches


def predict_rating_for_movie(user_id, movie_id):
    try:
        prediction = final_model.predict(user_id, movie_id)
        return prediction
    except Exception as e:
        return None



def display_menu():
    print("\n" + "-" * 60)
    print("MENU:")
    print("1. Get 10 best movie recommendations")
    print("2. Search for a movie and get rating prediction")
    print("3. Exit")
    print("-" * 60)


def main_interactive_loop():
    # Ask for user ID at startup
    print("\n" + "=" * 60)
    user_input = input(f"Enter your User ID (press Enter for new user [{new_user_id}]): ").strip()

    if user_input == "":
        current_user_id = new_user_id
        print(f"Using new user ID: {current_user_id}")
    else:
        try:
            current_user_id = int(user_input)
            print(f"Using User ID: {current_user_id}")
        except ValueError:
            print(f"Invalid input. Using new user ID: {new_user_id}")
            current_user_id = new_user_id

    # Main loop
    while True:
        display_menu()
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            # Get top 10 recommendations
            print("\n" + "=" * 60)
            print(f"TOP 10 MOVIE RECOMMENDATIONS FOR USER {current_user_id}")
            print("=" * 60)

            recommendations = get_top_n_recommendations(current_user_id, n=10)

            if recommendations:
                for i, (movie_id, pred_rating, title, genres) in enumerate(recommendations, 1):
                    print(f"\n{i}. {title}")
                    print(f"   Predicted Rating: {pred_rating:.2f}/5.0")
                    print(f"   Genres: {genres}")
            else:
                print("No recommendations available.")

        elif choice == "2":
            # Search and predict rating
            print("\n" + "=" * 60)
            print("SEARCH FOR A MOVIE AND GET RATING PREDICTION")
            print("=" * 60)

            search_query = input("\nEnter movie title to search: ").strip()

            if search_query:
                matches = search_movie(search_query)

                if len(matches) == 0:
                    print(f"No movies found matching '{search_query}'")
                else:
                    print(f"\nFound {len(matches)} movie(s):")
                    print("-" * 60)

                    for idx, row in matches.iterrows():
                        print(f"\nMovie ID: {row['movieId']}")
                        print(f"Title: {row['title']}")
                        print(f"Genres: {row['genres']}")
                        print(f"Year: {row['year']}")

                    # Ask user to select a movie
                    movie_id_input = input("\nEnter Movie ID to get prediction (or press Enter to cancel): ").strip()

                    if movie_id_input:
                        try:
                            selected_movie_id = int(movie_id_input)

                            # Check if movie exists
                            if selected_movie_id in matches['movieId'].values:
                                prediction = predict_rating_for_movie(current_user_id, selected_movie_id)

                                if prediction:
                                    movie_title = matches[matches['movieId'] == selected_movie_id].iloc[0]['title']
                                    pred_rating = prediction.est

                                    print("\n" + "=" * 60)
                                    print(f"PREDICTION FOR: {movie_title}")
                                    print("=" * 60)
                                    print(f"Predicted Rating: {pred_rating:.2f}/5.0")

                                    # Message about how likely they will enjoy it
                                    if pred_rating >= 4.5:
                                        print("🌟 You will LOVE this movie!")
                                    elif pred_rating >= 4.0:
                                        print("😊 You will likely really enjoy this movie!")
                                    elif pred_rating >= 3.5:
                                        print("👍 You will probably like this movie.")
                                    elif pred_rating >= 3.0:
                                        print("🤔 You might find this movie okay.")
                                    else:
                                        print("😕 You may not enjoy this movie much.")
                                else:
                                    print("Could not generate prediction for this movie.")
                            else:
                                print("Invalid Movie ID.")
                        except ValueError:
                            print("Invalid input.")

        elif choice == "3":
            # Exit
            print("\n" + "=" * 60)
            print("Thank you for using the Movie Recommendation System!")
            print("=" * 60)
            break

        else:
            print("\nInvalid option. Please select 1-3.")


# Start the interactive prototype
if __name__ == "__main__":
    main_interactive_loop()

