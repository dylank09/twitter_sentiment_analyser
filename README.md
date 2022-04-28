# Twitter Sentiment Analyser

I created a Keras Sequential model including an embedding layer and trained it on [this dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) containing 1.6 million tweets

The model achieved approx. 80% accuracy on the dataset.

I then saved the model and the tokenizer that I created and fitted on the data

The python script `twitter_sentiment_analyser.py` reads in the model and the tokenizer and uses them to predict the sentiment of a given Twitter search term

The script uses the twitter api to search for the given search term and returns the most recent and most popular 1000 tweets relevant to the search term.

The output of the script is a an object:
```
    result: {
        averageScore,
        totalPreds,
        countPositives,
        countNegatives,
        countNeutral
    }
```
It also prints out the word "POSITIVE", "NEGATIVE", or "NEUTRAL" based on the percentage of positive sentiment predictions

### How to use

1. Download this script (you will need access to the model.h5 and tokenizer.pkl files)
2. Run the following command on the command line, inputting your chosen search term instead of the curly braces
   `python twitter_sentiment_analyser.py -search {SEARCHTERM}`
3. Wait until the code is finished executing and observe the result
