import pandas as pd
import numpy as np
from ast import literal_eval

metadata = pd.read_csv("../data/movies_metadata.csv",low_memory=False)
ratings = pd.read_csv("../data/ratings_small.csv")
credits = pd.read_csv("../data/credits.csv")
keywords = pd.read_csv("../data/keywords.csv")
emotions_to_genres = pd.read_csv("../data/emotion_genre.csv")


metadata = metadata.iloc[0:10000,:]
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')
metadata.shape

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
  metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
  
  
  
#Getting a list of the actors, keywords and genres
def get_list(x):
    if isinstance(x, list): #checking to see if the input is a list or not
        names = [i['name'] for i in x] #if we take a look at the data, we find that
        #the word 'name' is used as a key for the names actors, 
        #the actual keywords and the actual genres
        
        #Check if more than 3 elements exist. If yes, return only first three. 
        #If no, return entire list. Too many elements would slow down our algorithm 
        #too much, and three should be more than enough for good recommendations.
        if len(names) > 3:
            names = names[:3]
        return names
    return []
  

  

metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

metadata[['title', 'cast', 'director', 'keywords', 'genres']].head()

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x] #cleaning up spaces in the data
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
          
         
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
  metadata[feature] = metadata[feature].apply(clean_data)

metadata.head()

def aggregate_emotions(genres_list, emotions_to_genres_df):
    all_emotions = []
    for genre in genres_list:
        # Find the matching emotions for the genre
        matched_emotions = emotions_to_genres_df[emotions_to_genres_df['genres'] == genre]['emotion'].tolist()
        all_emotions.extend(matched_emotions)
    
    # Aggregate emotions by concatenating them, or you can choose other methods of aggregation
    aggregated_emotions = ' '.join(all_emotions)
    return aggregated_emotions

metadata['emotions'] = metadata['genres'].apply(lambda x: aggregate_emotions(x, emotions_to_genres))

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) + ' ' + x['emotions']

metadata['soup'] = metadata.apply(create_soup, axis=1)
#print(metadata.columns)

metadata[['title', 'soup', 'cast', 'director', 'keywords', 'genres','emotions']].head()