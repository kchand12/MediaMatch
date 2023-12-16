# CourseProject

[Overview of Function Code]

This code is part of a chatbot application that can be used for movie recommendations using python. The application utilizes natural language processing for understanding user inputs and a recommendation system for suggesting movies based on user preferences. NLP processing utilizes 'nltk' and a custom trained model using Tensorflow to parse and classify user input into different categories or intents.
The movie recommender system includes functionality to recommend movies based on genres, keywords, emotions, actors and directors. A usage scenario of this application would look like a user interacts with the chatbot expressing their preference of movies. The chatbot processes the input classifying it into different categories and will ask follow-up questions. Once all the user information is gathered the chatbot will use the movie recommendation system to print out some movie suggestions.

[Documentation of Software Implementation]

Key Modules used:
'nltk': This is used for tokenizing and processing natural langauge data.
'TensorFlow': Helps build and run machine learning model
'tflearn': High-level API for Tensorflow that is used for building neural network for NLP
'pandas': This module is used in the manipulation and anlysis for handling movie datasets
'json': Parse JSON files that load chatbot intents and movie data 

The recommender system uses Term Frequency-Inverse Document Frequency. This is a statitic that represents how important a word is in a document. It also used cosine similarity to measure how similar two words are. The main function takes in user input in the form of search terms, vectorizes using TF-IDF and then computes the cosine similarity between the vecotrized input and the TF-IDF matrix. Thn the top N similar movies are selected based on the similarity scores.

[Installation and Running]

Ensure all the dependenies outlined in the requirements.txt are installed. IMPORTANT: please download all the datasets found here (https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and store in the data file. Both credits.csv and movies_metadata.csv were too big to upload to github so the current repository only reflect smaller snippets. 
This application is using Streamlit fronend to creat an interactive web application where users can input their preferences. Please install streamlit and navigate to the src folder. From there run the command: streamlit run chatbot_app.py
