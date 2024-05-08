import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk

# Load the dataset
df = pd.read_csv("movie_dataset.csv")
features = ['keywords', 'cast', 'genres', 'director']

# Function to combine features
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]

# Fill missing values
for feature in features:
    df[feature] = df[feature].fillna('')

# Create combined features column
df["combined_features"] = df.apply(combine_features, axis=1)

# Vectorize features
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# Functions to get title from index and index from title
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

# Function to recommend movies
def recommend_movies(movie_user_likes):
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
    top_similar_movies = sorted_similar_movies[:15]
    recommended_movies = [get_title_from_index(movie[0]) for movie in top_similar_movies]
    return recommended_movies

# Function to handle button click
def on_button_click():
    movie_user_likes = entry.get()
    recommended_movies = recommend_movies(movie_user_likes)
    recommended_movies_text.set("\n".join(recommended_movies))

# GUI setup
root = tk.Tk()
root.title("Movie Recommendation System")

# Increase font size of GUI
root.option_add("*Font", "Century 12")

label = tk.Label(root, text="Enter a movie:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

# Add "Enter" button
button = tk.Button(root, text="Recommend", command=on_button_click)
button.pack()

recommended_movies_text = tk.StringVar()
recommended_movies_label = tk.Label(root, textvariable=recommended_movies_text)
recommended_movies_label.pack()

root.mainloop()
