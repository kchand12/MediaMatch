def get_genres():
  genres = input("What Movie Genre are you interested in (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  genres = " ".join(["".join(n.split()) for n in genres.lower().split(',')])
  return genres

def get_keywords():
  keywords = input("What are some of the keywords that describe the movie you want to watch, like elements of the plot, whether or not it is about friendship, etc? (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  keywords = " ".join(["".join(n.split()) for n in keywords.lower().split(',')])
  return keywords

def get_emotion():
    emotions = input("What emotions would you like the movie to evoke or explore? ex. emotional, intense, exciting, passionate, scary etc. (if multiple, please separate them with a comma) [Type 'skip' to skip this question] ")
    emotions = " ".join(["".join(e.strip()) for e in emotions.lower().split(',')])
    return emotions

def get_actors():
  actors = input("Who are some actors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  actors = " ".join(["".join(n.split()) for n in actors.lower().split(',')])
  return actors

def get_directors():
  directors = input("Who are some directors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  directors = " ".join(["".join(n.split()) for n in directors.lower().split(',')])
  return directors



def get_searchTerms():
  searchTerms = [] 
  genres = get_genres()
  if genres != 'skip':
    searchTerms.append(genres)

  keywords = get_keywords()
  if keywords != 'skip':
    searchTerms.append(keywords)
  
  emotion = get_emotion()
  if emotion != 'skip':
    searchTerms.append(emotion)

  actors = get_actors()
  if actors != 'skip':
    searchTerms.append(actors)

  directors = get_directors()
  if directors != 'skip':
    searchTerms.append(directors)

  

  return searchTerms