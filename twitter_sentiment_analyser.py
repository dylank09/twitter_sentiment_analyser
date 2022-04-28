from pyexpat import model
from matplotlib.pyplot import get
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import pickle
import time
import os
import sys
import tweepy as tw

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.6)

# TWITTER
TWITTER_THRESHOLDS = (0.4, 0.6)

#KERAS
SEQUENCE_LENGTH = 300

#KEYS
API_KEY = "YStExELEbrq96K91N9yqG7lFK"
API_KEY_SECRET = "mrmgAe9Kv8C8iMJB4XyLAIKhqUbcVru7xL3VspUqqr563R3s8v"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALzgbwEAAAAAwmU8xps7fkx78zFGydUawYiJfqA%3DU1gBCiFbFDB8T0PKblt6uwc63yvMA5tfQ0mGhQdUXHadiLKh8I"

SEARCH_TERM = "#elonmusk"

model = load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score), "elapsed_time": time.time()-start_at}  

def get_tweets():
    auth = tw.OAuth2BearerHandler(BEARER_TOKEN)
    api = tw.API(auth, wait_on_rate_limit=True)
    tweets = tw.Cursor(api.search_tweets, q=SEARCH_TERM, lang="en").items(1000)
    return [tweet.text for tweet in tweets]

def make_predictions(tweet_texts):
    predictions = []
    total_score = 0
    count_positive = 0
    count_negative = 0
    count_neutral = 0

    for tweet_text in tweet_texts:
        prediction = predict(tweet_text)
        predictions.append(prediction)
        total_score += prediction['score']
        
        if prediction['label'] == 'POSITIVE':
            count_positive += 1
        elif prediction['label'] == 'NEGATIVE':
            count_negative += 1
        else:
            count_neutral += 1
            
    total_preds = len(predictions)
    average_score = total_score / total_preds
    return {"averageScore": average_score, 
            "totalPreds": total_preds,
            "countPositives": count_positive,
            "countNegatives": count_negative,
            "countNeutral": count_neutral}

def get_overall_sentiment(positive_percent):
    overall_sentiment = "NEUTRAL"

    if positive_percent > TWITTER_THRESHOLDS[1]:
        overall_sentiment = "POSITIVE"
    elif positive_percent < TWITTER_THRESHOLDS[0]:
        overall_sentiment = "NEGATIVE"
        
    return overall_sentiment

def main():
        
    args = sys.argv[1:]
    
    if len(args) == 2 and args[0] == '-search':
        SEARCH_TERM = args[1]
        print(SEARCH_TERM)

    tweet_texts = get_tweets()
    result = make_predictions(tweet_texts)
    print(result)
    positive_percent = result["countPositives"] / result["totalPreds"]
    overall_sent = get_overall_sentiment(positive_percent)
    print(overall_sent)

if __name__ == "__main__":
    main()