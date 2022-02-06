
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import bs4 as bs
import urllib.request
import nltk
import numpy as np
import random
nltk.download('punkt')
nltk.download('wordnet')

# scrape train data from wiki
blackHole_data = urllib.request.urlopen(
    'https://en.wikipedia.org/wiki/Black_hole').read()

blackHole_data_paragraphs = bs.BeautifulSoup(
    blackHole_data, 'lxml').find_all('p')
# Creating the corpus of all the web page paragraphs

blackHole_text = ''
# Creating lower text corpus of black holes paragraphs
for p in blackHole_data_paragraphs:
    blackHole_text += p.text.lower()
print(blackHole_text)

blackHole_text = re.sub(
    r'\s+', ' ', re.sub(r'\[[0-9]*\]', ' ', blackHole_text))

sent_tokens = nltk.sent_tokenize(blackHole_text)
sent_tokens[:4]


GREETING_INPUTS = ("hello", "hi", "greetings", "hey there!", "", "hey",)

GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "nice to meet you!"]


def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def astro_chat(user_response):
    astro_response = ''
    sent_tokens.append(user_response)

# running tfidf
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(sent_tokens)
    # cosine similarity taking the second closest index since the first is user response
    cosine_vectors = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)
    idx = cosine_vectors.argsort()[0][-2]
    flat = cosine_vectors.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        astro_response = astro_response + \
            "I don't seem to have information on that, sorry. I love talking about black holes though - my favourite topic for the week! "
        return astro_response
    else:
        astro_response = astro_response+sent_tokens[idx]
        return astro_response


flag = True
print("ASTRO: Hey, my name is astro! If there's anything you'd like to know about black holes -- my favourite topic of the week! ")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == "thank you"):
            flag = False
            print("ASTRO: You're welcome! There's so much more to explore in space!")
        else:
            if(greeting(user_response) != None):
                print("ASTRO: "+greeting(user_response))
            else:
                print("ASTRO: ", end="")
                print(astro_chat(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ASTRO: Bye! Can't wait to explore space with you again!")
