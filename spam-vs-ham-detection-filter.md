*Full project with complete code and datasets available on my Github [repository](https://github.com/ruthgn/Movie-Recommendation-System).*

With the ubiquity of mobile phone devices expanding by the day, Short Message
Service (SMS) or "texting" remains one of the most broadly utilized communication services. In any case, this has prompted an expansion in mobile phones attacks like SMS Spam. Fortunately, using natural language processing (NLP) concepts, we can train machines to conveniently remove these unsolicited messages for us. 

NLP combines machine learning techniques with text by using math and statistics to get that text in a format that the machine learning algorithms can understand. This post lays that exact process out using Python's NLTK (Natural Language Toolkit) library. NLTK has a lot of useful features and is widely considered to be the standard library for processing text in Python. 

We'll be using a dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) storing a collection of more than five thousand SMS text messages. Our goal is to build a spam detection filter that performs a classification task to separate text messages that are in fact spam versus those that are legitimate text messages sent by real people. Using these labeled ham and spam examples, we'll **train a machine learning model to learn to discriminate between ham/spam automatically**. Then, with a trained model, we'll be able to **classify arbitrary unlabeled messages** as ham or spam.

## Getting Started


```python
# Import NLTK library
import nltk
```

Note: The next step requires some manual input through NLTK's interactive shell.


```python
# Download Stopwords Corpus package
# nltk.download_shell()
```

Let's go ahead and check out the data (i.e., read random anonymous text messages)!


```python
# Use rstrip() plus a list comprehension 
# to get a list of all the lines of text messages
messages = [line.rstrip() for line in open ('smsspamcollection/SMSSpamCollection')]
```


```python
# Number of messages in the dataset
print(len(messages))
```

    5574
    


```python
# Check out a random text(message no. 100) with its spam/ham status
messages[99]
```




    'ham\tI see a cup of coffee animation'




```python
# Another one
messages[42]
```




    'spam\t07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow'




```python
# Print first ten messages on the list and number them using enumerate
for msgNo, message in enumerate(messages[:10]):
    print(msgNo, message)
    print('\n')
```

    0 ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    
    
    1 ham	Ok lar... Joking wif u oni...
    
    
    2 spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
    
    
    3 ham	U dun say so early hor... U c already then say...
    
    
    4 ham	Nah I don't think he goes to usf, he lives around here though
    
    
    5 spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv
    
    
    6 ham	Even my brother is not like to speak with me. They treat me like aids patent.
    
    
    7 ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
    
    
    8 spam	WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
    
    
    9 spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
    
    
    

From just a preliminary view, we can already see the style of these text messages, particularly those that are labelled spam. These spam messages seem to be your standard spam mails asking you for money or claiming that you're a winner of something. What we want to do is to actually figure out how we can detect which text messages are "spam" versus which ones are "ham" (normal text messages).

Due to the spacing we can almost immediately tell that this is a [TSV](http://en.wikipedia.org/wiki/Tab-separated_values) ("tab separated values") file, where the first column is a label saying whether the given message is a normal message ("ham") or "spam", while the second column is the message itself (note that our numbers aren't part of the file, they are just from the **enumerate** call). Instead of parsing TSV manually using Python, we can automate the process using pandas!


```python
# Import pandas
import pandas as pd
```


```python
# Read data
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', 
                       sep='\t', names=['label', 'message'])
```


```python
# Quick look at data
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis

Let's check out the stats with some plots and pandas built-in methods!


```python
# Descriptive statistics
messages.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5572</td>
      <td>5572</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>5169</td>
    </tr>
    <tr>
      <th>top</th>
      <td>ham</td>
      <td>Sorry, I'll call later</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4825</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages.groupby('label').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">message</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4825</td>
      <td>4516</td>
      <td>Sorry, I'll call later</td>
      <td>30</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
      <td>653</td>
      <td>Please call our customer service representativ...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



As we continue our analysis we want to start thinking about the features we are going to be using. This goes along with the general idea of [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)--which is a very large part of spam detection and natural language processing in general. The better our domain knowledge on the data, the better our ability to engineer more features from it.


```python
# Detecting length of text messages and adding a length column
messages['length'] = messages['message'].apply(len)
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



Let's create some visualizations!


```python
# Import data visualization libraries and set style
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
```


```python
# Visualize message frequency vs. length
sns.distplot(messages['length'], kde=False, bins=70)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b0bfdf3b48>




![png](spam-vs-ham-detection-filter_files/spam-vs-ham-detection-filter_25_1.png)


Looks like text length may be a good feature to think about! Let's try to investigate why the x-axis goes all the way to around one thousand--this must mean that there is some really long message.


```python
# Length data descriptive statistics
messages['length'].describe()
```




    count    5572.000000
    mean       80.489950
    std        59.942907
    min         2.000000
    25%        36.000000
    50%        62.000000
    75%       122.000000
    max       910.000000
    Name: length, dtype: float64




```python
# Locate the longest message
messages[messages['length']==910]['message'].iloc[0]
```




    "For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.she should be there every time I dream.love will be then when my every breath has her name.my life should happen around her.my life will be named to her.I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later.."



Mmmmm....okay.

But let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:


```python
messages.hist(column='length', by='label', bins=70, figsize=(12,6))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000001B0BFE10B08>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x000001B0BFD91F88>],
          dtype=object)




![png](spam-vs-ham-detection-filter_files/spam-vs-ham-detection-filter_31_1.png)


Fantastic! At this point we can safely assume that message length is a good feature to distinguish 'spam' vs 'ham' messages. Text messages that are spam tend to be longer than those that are ham. Through our exploratory data analysis we discovered a trend that spam messages tend to have more characters (unless it's a love letter!).

We will now begin to process the data to eventually use with SciKit Learn.

## Text pre-processing

There are numerous methods to convert a corpus to a numerical feature vector for machine learning algorithms to perform classification task on; one of which is the the [bag-of-words](http://en.wikipedia.org/wiki/Bag-of-words_model) approach, where each unique word in a text will be represented by one number.

In this section, we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers). This step includes removing punctuations, splitting a message into its individual words, and removing very common words (e.g., "the", "a", "be", "me", "is", etc.).


```python
# Remove punctuations

# Use python's built-in 'string' library to list all the possible punctuation
import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string
nopunc = ''.join(nopunc)
```


```python
# Split a message into a list of words
nopunc.split()
```




    ['Sample', 'message', 'Notice', 'it', 'has', 'punctuation']




```python
# Import a list of english stopwords from NLTK
from nltk.corpus import stopwords

# Display examples of some stop words
stopwords.words('english')[0:20] 
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his']




```python
# Remove stop words
cleanMsg = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
cleanMsg
```




    ['Sample', 'message', 'Notice', 'punctuation']




```python
# Put all preprocessing steps in a function to apply to our DataFrame later on
def textProcess(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
```


```python
# Display original DataFrame
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>



What we need to do is to "tokenize" these messages. Tokenization is the term used to describe the process of converting a normal text string into a list of tokens (words that we want to have for analysis).


```python
# Apply function for tokenization
messages['message'].head(5).apply(textProcess)
```




    0    [Go, jurong, point, crazy, Available, bugis, n...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
    3        [U, dun, say, early, hor, U, c, already, say]
    4    [Nah, dont, think, goes, usf, lives, around, t...
    Name: message, dtype: object



*Note*: There are many ways to normalize text data. The NLTK library contains numerous built-in tools and great documentation on other methods of normalization. We're going to focus on using what we have to convert our list of words to an actual vector that SciKit-Learn can use.

## Vectorization

At this point, we have the messages as lists of tokens (also known as [lemmas](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) which we need to convert into a vector form that SciKit Learn's algorithm models can work with.

We're going to convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.

To summarize the three steps using the bag-of-words model:

1. Count how many times does a word occur in each message (Known as term frequency)

2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)

3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

Let's begin the first step:

Each vector will have as many dimensions as there are unique words in the SMS corpus.  We will first use SciKit Learn's **CountVectorizer**. This model will convert a collection of text documents to a matrix of token counts.

We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (1 row per word) and the other dimension are the actual documents, in this case a column per text message. 

For example:

<table border = “1“>
<tr>
<th></th> <th>Message 1</th> <th>Message 2</th> <th>...</th> <th>Message N</th> 
</tr>
<tr>
<td><b>Word 1 Count</b></td><td>0</td><td>1</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>Word 2 Count</b></td><td>0</td><td>0</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>...</b></td> <td>1</td><td>2</td><td>...</td><td>0</td>
</tr>
<tr>
<td><b>Word N Count</b></td> <td>0</td><td>1</td><td>...</td><td>1</td>
</tr>
</table>


Since there are so many messages, we can expect a lot of zero counts for the presence of that word in that document. Because of this, SciKit Learn will output a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix).


```python
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
```

There are a lot of arguments and parameters that can be passed to the CountVectorizer. In this case we will just specify the **analyzer** to be our own previously defined function:


```python
# Specify textProcess as analyzer. Warning: This might take a while
bowTransformer = CountVectorizer(analyzer=textProcess).fit(messages['message'])
```


```python
# Print total number of vocabularies/words
print(len(bowTransformer.vocabulary_))
```

    11425
    


```python
# Take a random text message to get its bag-of-words counts as a vector 
msg4 = messages['message'][3]
print(msg4)
```

    U dun say so early hor... U c already then say...
    


```python
# Use bow_transformer
bow4 = bowTransformer.transform([msg4])
```


```python
# Display vector representation of the text message
print(bow4)
print(bow4.shape)
```

      (0, 4068)	2
      (0, 4629)	1
      (0, 5261)	1
      (0, 6204)	1
      (0, 6222)	1
      (0, 7186)	1
      (0, 9554)	2
    (1, 11425)
    

This means that there are seven unique words in our selected message (after removing common stop words). Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:


```python
# Display the repeated words
print(bowTransformer.get_feature_names()[4068])
print(bowTransformer.get_feature_names()[9554])
```

    U
    say
    

Now we can use **.transform** on our Bag-of-Words (bow) transformer object and apply it on the entire DataFrame of messages. Check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix.


```python
# Transform entire DataFrame of messages
messagesBow = bowTransformer.transform(messages['message'])
```


```python
# Display bag-of-words counts for the entire SMS corpus
print('Shape of Sparse Matrix: ', messagesBow.shape)
# Checking non-zero occurences
print('Amount of Non-Zero occurences: ', messagesBow.nnz)
```

    Shape of Sparse Matrix:  (5572, 11425)
    Amount of Non-Zero occurences:  50548
    


```python
sparsity = (100.0) * messagesBow.nnz / (messagesBow.shape[0] * messagesBow.shape[1])
print('sparsity: {}'.format(sparsity))
```

    sparsity: 0.07940295412668218
    

After the counting, weighting and normalization can be done with [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf), using scikit-learn's `TfidfTransformer`.

TF-IDF stands for *term frequency-inverse document frequency*, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.

One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.

Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), meaning the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

**TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 

*TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*

**IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 

*IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*

Below is a simple example:

Consider a document containing 100 words wherein the word cat appears 3 times. 

The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.

Let's go ahead and do this in SciKit Learn:


```python
# Import scikit-learn's TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer

# Calculate our single random message bag-of-words TF-IDF weights
tfidfTransformer = TfidfTransformer().fit(messagesBow)
tfidf4 = tfidfTransformer.transform(bow4)
print(tfidf4)
```

      (0, 9554)	0.5385626262927564
      (0, 7186)	0.4389365653379857
      (0, 6222)	0.3187216892949149
      (0, 6204)	0.29953799723697416
      (0, 5261)	0.29729957405868723
      (0, 4629)	0.26619801906087187
      (0, 4068)	0.40832589933384067
    


```python
# Check the IDF (inverse document frequency) of the word "u"
print(tfidfTransformer.idf_[bowTransformer.vocabulary_['u']])
# Check the IDF (inverse document frequency) of the word "university"
print(tfidfTransformer.idf_[bowTransformer.vocabulary_['university']])
```

    3.2800524267409408
    8.527076498901426
    


```python
# Transform the entire bag-of-words corpus into TF-IDF corpus at once
messagesTfidf = tfidfTransformer.transform(messagesBow)
```

*Note*: As is the case with how text data can be preprocessed, there are many ways data can be vectorized. These steps involve feature engineering and building a "pipeline". SciKit Learn's documentation on dealing with text data as well as the expansive collection of books on the general topic of NLP are tremendous resources everyone needs to check out.

## Training our model

With messages represented as vectors, we can finally train our spam/ham classifier using some sort of classification algorithm. We will be using the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier algorithm with scikit-learn.


```python
from sklearn.naive_bayes import MultinomialNB
spamDetectModel = MultinomialNB().fit(messagesTfidf, messages['label'])
```

Let's try classifying our single random message and checking how we do:


```python
# Let's try classifying our single random message and checking how we do:
print('predicted:', spamDetectModel.predict(tfidf4)[0])
print('expected:', messages.label[3])
```

    predicted: ham
    expected: ham
    

Voila!! We've developed a model that can attempt to predict spam vs ham classification!


```python
spamDetectModel.predict(tfidf4)[0]
```




    'ham'



## Model Evaluation

Now let's determine how well our model will do overall on the entire dataset. We'll begin by getting all the predictions:


```python
messages['label'][98]
```




    'ham'




```python
# Get all predictions
pred = spamDetectModel.predict(messagesTfidf)
print(pred)
```

    ['ham' 'ham' 'spam' ... 'ham' 'ham' 'ham']
    

We can use SciKit Learn's built-in classification report, which returns [precision, recall,](https://en.wikipedia.org/wiki/Precision_and_recall) [f1-score](https://en.wikipedia.org/wiki/F1_score), and a column for support (meaning how many cases supported that classification).


```python
# Import classification_report
from sklearn.metrics import classification_report

# Display model "evaluation"
print (classification_report(messages['label'], pred))
```

                  precision    recall  f1-score   support
    
             ham       0.98      1.00      0.99      4825
            spam       1.00      0.85      0.92       747
    
        accuracy                           0.98      5572
       macro avg       0.99      0.92      0.95      5572
    weighted avg       0.98      0.98      0.98      5572
    
    

There are quite a few possible metrics for evaluating model performance. We can assume, for example, that the cost of mis-predicting "spam" as "ham" is probably much lower than mis-predicting "ham" as "spam".

In the above "evaluation",we evaluated accuracy of our trained model on the same data we used for training. **In reality we should never actually evaluate on the same dataset we train our model on!**. Such evaluation tells us nothing about the true predictive power of our model. If we simply remembered each example during training, the accuracy on training data would trivially be 100%, even though we wouldn't be able to classify any new messages.

The proper way to proceed is to add a step where we split the data into a training/test set, where the model only ever sees the **training data** during its model fitting and parameter tuning. The **test data** is never used in any way. This is then our final evaluation on test data is representative of true predictive performance.

## Train Test Split


```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the data
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.3)

# Display train/test split
print('Training Set:', len(msg_train))
print('Test Set:', len(msg_test))
print('Total:', len(msg_train) + len(msg_test))
```

    Training Set: 3900
    Test Set: 1672
    Total: 5572
    

## Creating a Data Pipeline

Let's run our model again and then predict off the test set. We will use SciKit Learn's [pipeline](http://scikit-learn.org/stable/modules/pipeline.html) capabilities to store a pipeline of workflow. This will allow us to set up all the transformations that we will do to the data for future use.


```python
# Import pipeline
from sklearn.pipeline import Pipeline
```


```python
# Store workflow into pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=textProcess)), # strings to token integer counts
    ('tfidf', TfidfTransformer()), # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()) # train on TF-IDF vectors w/ Naive Bayes classifier
])
```

Now we can directly pass data from the text messages and the pipeline will do our pre-processing for us! We can treat it as a model/estimator API:


```python
pipeline.fit(msg_train, label_train)
```




    Pipeline(memory=None,
             steps=[('bow',
                     CountVectorizer(analyzer=<function textProcess at 0x000001B0C10C8828>,
                                     binary=False, decode_error='strict',
                                     dtype=<class 'numpy.int64'>, encoding='utf-8',
                                     input='content', lowercase=True, max_df=1.0,
                                     max_features=None, min_df=1,
                                     ngram_range=(1, 1), preprocessor=None,
                                     stop_words=None, strip_accents=None,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, vocabulary=None)),
                    ('tfidf',
                     TfidfTransformer(norm='l2', smooth_idf=True,
                                      sublinear_tf=False, use_idf=True)),
                    ('classifier',
                     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],
             verbose=False)




```python
predictions = pipeline.predict(msg_test)
```


```python
# Display model evaluation
print(classification_report(label_test, predictions))
```

                  precision    recall  f1-score   support
    
             ham       0.95      1.00      0.98      1443
            spam       1.00      0.68      0.81       229
    
        accuracy                           0.96      1672
       macro avg       0.98      0.84      0.89      1672
    weighted avg       0.96      0.96      0.95      1672
    
    

Now we have a classification report for our model on a true testing set! There is a lot more to Natural Language Processing than what we've covered here, and its vast expanse of topic could fill up several college courses!
