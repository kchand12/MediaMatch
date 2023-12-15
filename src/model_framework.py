import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
#import tflearn
import tensorflow as tf
import random
import json
import pickle
from recommender import genre_recommendations, make_recommendation

with open('../data/intent.json') as json_data:
    intents = json.load(json_data)

# words = []
# classes = []
# documents = []
# ignore_words = ['?']
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = nltk.word_tokenize(pattern)
#         # add to our words list
#         words.extend(w)
#         # add to documents in our corpus
#         documents.append((w, intent['tag']))
#         # add to our classes list
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # stem and lower each word and remove duplicates
# words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))

# # remove duplicates
# classes = sorted(list(set(classes)))

# training = []
# output = []
# output_empty = [0] * len(classes)

# for doc in documents:
#     # initialize our bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = doc[0]
#     # stem each word
#     pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
#     # create our bag of words array
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)

#     # output is a '0' for each tag and '1' for current tag
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1

#     training.append([bag, output_row])

# # shuffle our features and turn into np.array
# random.shuffle(training)
# training = np.array(training)

# # create train and test lists
# train_x = list(training[:,0])
# train_y = list(training[:,1])

# tf.compat.v1.reset_default_graph()
# # Build neural network
# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tflearn.regression(net)

# # Define model and setup tensorboard
# model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# # Start training (apply gradient descent algorithm)
# model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
# model.save('model.tflearn')

# pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# data = pickle.load( open( "training_data", "rb" ) )
# words = data['words']
# classes = data['classes']
# train_x = data['train_x']
# train_y = data['train_y']

# # import our chat-bot intents file
# import json
# with open('../data/intent.json') as json_data:
#     intents = json.load(json_data)

# model.load('./model.tflearn')

# def clean_up_sentence(sentence):
#     # tokenize the pattern
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word
#     sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
#     return sentence_words

# # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
# def bow(sentence, words, show_details=False):
#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # bag of words
#     bag = [0]*len(words)  
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s: 
#                 bag[i] = 1
#                 if show_details:
#                     print ("found in bag: %s" % w)

#     return(np.array(bag))

# context = {}

# ERROR_THRESHOLD = 0.25
# def classify(sentence):
#     # generate probabilities from the model
#     results = model.predict([bow(sentence, words)])[0]
#     # filter out predictions below a threshold
#     results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append((classes[r[0]], r[1]))
#     # return tuple of intent and probability
#     return return_list
def extract_genre(sentence):
    # Define a list of possible genres and keywords associated with them
    genres = {
        "comedy": ["comedy", "comedies", "funny", "humor"],
        "horror": ["horror", "scary", "frightening", "terror"],
        "sci-fi": ["sci-fi", "science fiction", "sci fi", "space", "futuristic"],
        "romance": ["romance", "romantic", "love", "lovestory", "love story"],
        "action": ["action", "fast-paced", "thrill", "exciting", "adventure"],
        "animation": ["animation", "animated", "cartoon", "anime"],
        "thriller": ["thriller", "suspense", "tense", "mystery"],
        "documentary": ["documentary", "docu", "real life", "true story"],
        "drama": ["drama", "dramatic", "serious", "emotional"],
        "adventure": ["adventure", "explore", "exploration", "journey"],
        "history": ["history", "historical", "past", "period"],
        "fantasy": ["fantasy", "magical", "fairy tale", "mythical"]
        # Add more genres and their keywords as needed
    }

    # Tokenize the sentence and convert to lower case
    words = sentence.lower().split()
    # Check each word in the sentence to see if it matches a genre keyword
    for word in words:
        for genre, keywords in genres.items():
            if word in keywords:
                return genre
    return None
def format_recommendations(recommendations):
    if recommendations:
        formatted = "Top movie recommendations:\n" + "\n".join([f"{title} (IMDb ID: {imdb_id})" for title, imdb_id in recommendations])
        return formatted
    else:
        return "No recommendations available at the moment."

def response(sentence, userID='123', show_details=False, context={}):
    # Initialize state if not present in context
    if 'state' not in context:
        context['state'] = 'start'

    # Process input based on the current state
    if context['state'] == 'start':
        context['state'] = 'ask_genre'
        return "What movie genre are you interested in?", context

    elif context['state'] == 'ask_genre':
        context['genres'] = sentence
        context['state'] = 'ask_keywords'
        return "What are some keywords that describe the movie you want to watch?", context

    elif context['state'] == 'ask_keywords':
        context['keywords'] = sentence
        context['state'] = 'ask_emotion'
        return "What emotions would you like the movie to evoke?", context

    elif context['state'] == 'ask_emotion':
        context['emotion'] = sentence
        context['state'] = 'ask_actors'
        return "Who are some actors within the genre that you love?", context

    elif context['state'] == 'ask_actors':
        context['actors'] = sentence
        context['state'] = 'ask_directors'
        return "Who are some directors within the genre that you love?", context

    elif context['state'] == 'ask_directors':
        context['directors'] = sentence
        context['state'] = 'completed'
        search_terms = [context.get(key) for key in ['genres', 'keywords', 'emotion', 'actors', 'directors']]
        search_terms = [term for term in search_terms if term]  # Filter out any 'None' values
        recommendations = make_recommendation(search_terms=search_terms)
        recommendations_str = format_recommendations(recommendations)
        context['state'] = 'start'  # Reset the state for a new conversation
        return recommendations_str, context

    # Reset state and return a default response for any other intents
    context['state'] = 'start'
    return "I'm not sure how to respond to that.", context
# def response(sentence, userID='123', show_details=False, context={}):    
#   results = classify(sentence)
#     if results:
#         while results:
#             for i in intents['intents']:
#                 if i['tag'] == results[0][0]:
#                     if 'context_set' in i:
#                         context[userID] = i['context_set']
#                         if show_details: 
#                             print('context set to:', i['context_set'])

#                     if not 'context_filter' in i or (context.get(userID) == i['context_filter']):
#                         if i['tag'] == 'movie_recommendation' or i['tag'] == 'no_preference':
#                             recommendations = make_recommendation()
#                             recommendations_str = format_recommendations(recommendations)
#                             return recommendations_str
#                         if i['tag'] == 'specific_genre':
#                             genre = extract_genre(sentence)
#                             if genre:
#                                 genre_specific_recommendations = genre_recommendations(genre)
#                                 genre_str = format_recommendations(genre_specific_recommendations)
#                                 return genre_str
#                             else:
#                                 r = make_recommendation()
#                                 return format_recommendations(r)
#                         return random.choice(i['responses']), context
#             results.pop(0)
    return "I'm not sure how to respond to that.", context
