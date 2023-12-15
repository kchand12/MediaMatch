from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_clean import metadata
from user_input import get_searchTerms
import pandas as pd
import numpy as np

# Precompute the TF-IDF matrix for the existing dataset
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['soup'])
metadata = metadata

def make_recommendation(search_terms):
    # Get user input
    search_terms_str = " ".join(search_terms)
    # Vectorize the user input
    user_input_tfidf = tfidf.transform([search_terms_str])
    # search_terms = get_searchTerms()
    # search_terms_str = " ".join(search_terms)

    # Vectorize the user input
    user_input_tfidf = tfidf.transform([search_terms_str])

    # Compute cosine similarity between user input and the existing TF-IDF matrix
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)

    # Flatten the similarity scores array and get top 10 similar movies
    sim_scores = cosine_sim.flatten()
    top_indices = np.argsort(sim_scores)[::-1][1:11]  # Skip the first one as it will be the user's input

    # Retrieve the recommended movie titles and IMDb IDs
    recommended_titles = [(metadata['title'].iloc[i], metadata['imdb_id'].iloc[i]) for i in top_indices]
    #print(recommended_titles)
    return recommended_titles

def genre_recommendations(genre):
    num_recommendations=10
    # Filter the metadata for the specified genre
    metadata['genres'] = metadata['genres'].str.lower()
    genre_metadata = metadata[metadata['genres'].str.contains(genre, case=False, na=False)]

    # Check if there are enough movies in the specified genre
    if genre_metadata.shape[0] == 0:
        return f"No movies found in the genre: {genre}"

    # Compute TF-IDF matrix for the filtered metadata
    genre_tfidf_matrix = tfidf.transform(genre_metadata['soup'])

    # Use the mean of the TF-IDF vectors for the genre as the query vector
    query_vector = np.mean(genre_tfidf_matrix, axis=0)

    # Compute cosine similarity between the mean vector and the full TF-IDF matrix
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    # Flatten the similarity scores array and get top movies
    sim_scores = cosine_sim.flatten()
    top_indices = np.argsort(sim_scores)[::-1][:num_recommendations]

    # Retrieve the recommended movie titles
    recommended_titles = [(metadata['title'].iloc[i], metadata['imdb_id'].iloc[i]) for i in top_indices]
    return recommended_titles

# Let's try our updated recommendation function
#make_recommendation()

