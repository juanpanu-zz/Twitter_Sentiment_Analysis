# Twitter_Sentiment_Analysis

This project is aimed to create a tweet classifier. Where tweets with friendly content and tweets with inappropriate content are separated.

Libraries used:
  - Pandas
  - Matplotlib
  - Numpy
  - nltk
  - seaborn

There are two scripts, the sentiment analysis model is created in [TweetSense_Model](TweetSense_Model.ipynb) using a Naive Bayes multinomial classifier.

### Positive Tweets
Using pandas and nltk libraries i'm able to identify which are some of the positive words used in training dataset according to its label.
![Positive](/images/normalcloud.png)

### Negative Tweets
Using pandas and nltk libraries i'm able to identify which are some of the negative words used in training dataset according to its label.
![Negative](/images/negativecloud.png)

### Understanding impact of hashtags on tweet sentiment
With regular expressions I can identify the # character and save all the words that start with it and classify them acording to its label.
  #### positive hastags
![#positive](/images/positive_hastags.png)
  #### negative hastags
![#negative](/images/negative_hastags.png)

### Using Bag of words for converting data into features
```python

  #Bag-of-words
  #Each row in matrix M contains the frequency of tokens(words) in the document D(i)
  bow_vectorizer = CountVectorizer(max_df=0.90 ,min_df=2 , max_features=1000,stop_words='english')
  bow = bow_vectorizer.fit_transform(combine['tidy_tweet']) # tokenize and build vocabulary
  
```
### We will use Multinomial Naive Bayes Classifier
```python
  from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
  model_naive = MultinomialNB().fit(X_train, y_train) 
  predicted_naive = model_naive.predict(X_test)
  
```
### Model Metrics
![ConfusionMatrix](/images/confusion_matrix.png)
```python
  from sklearn.metrics import accuracy_score

  score_naive = accuracy_score(predicted_naive, y_test)
  print("Accuracy with Naive-bayes: ",score_naive)
  Accuracy with Naive-bayes:  0.9408055329536208
```
### Using Twitter API
API configuration and functions to convert gathered data to a dataframe is found in the second script [TweetSense_API](TweetSense_API.ipynb)##

### Twitter query
In this case the query is set to find 'racism' related tweets, this query will return a json with the information we want to analyse and will be transformed into a Dataframe.
```python

  # Twitter URL
  def create_twitter_url():
      handle = "racism"
      max_results = 100
      mrf = "max_results={}".format(max_results)
      q = "query={}".format(handle)
      url = "https://api.twitter.com/2/tweets/search/recent?{}&{}".format(
          mrf, q
      )
      return url

```
### Data Preprocessing Steps
These steps are basically the same as those used in the training stage.
### WordCloud
![Cloud](/images/WordCloud.png)

### Using NaiveBayes classifier
With the previously trained classifier, I can now classify positive or negative tweets.

#### Some Positive Tweets
![APIPositive](/images/Positive_API.png)
#### Some Negative Tweets
![APINegative](/images/Negative_API.png)
