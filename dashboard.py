import pandas as pd
import numpy as np
import re
import unidecode
import tweepy
import re
import contractions
#from pymongo.mongo_client import MongoClient
#from pymongo.server_api import ServerApi
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import matplotlib.pyplot as plt
import os
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Create a new client and connect to the server
#uri = "mongodb+srv://arpibibi01:YYmrMBOoPeVr0Ppo@era0.y55y42e.mongodb.net/?retryWrites=true&w=majority"
#mongo_client = MongoClient(uri, server_api=ServerApi('1'))



#Process Tweets
def clean_text(text):
    text = unidecode.unidecode(text)
    text = text.lower()

    # Removes all mentions (@userwrite) from the tweet
    text = re.sub(r'(@[A-Za-z0-9_]+)','', text)

    # Remove hashtags
    text = re.sub(r'(#[A-Za-z0-9_]+)','', text)

    # Removes any link in the text
    text = re.sub('http://\S+|https://\S+','', text)
    
    # Contract words
    #word contracions
    text = " ".join([contractions.fix(i) for i in text.split()])

    # Removes punctuation and leading whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()

    # Removes digits
    text = re.sub(r'[0-9]','',text)

    # Removes newline characters
    text = re.sub(r'\n','',text)

    return text

def model_load():
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    return tokenizer,model,labels

#tokenizer,model,labels=model_load()

def compute_sentiment(tokenizer,model,labels,review):
    k =[]
    encoded_input = tokenizer(review, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
    #s = scores[ranking[i]]
        k.append(l)
    s= k[0]
    return s
       

def data_extract(word,count):


     
    #bearer token for tweet api call
    tweet_client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAFS2oAEAAAAAA8uTf5I%2FNC80gvZJnMBYpULhiyM%3D118HttPeyhmRIO5OOmEXUaqXzCen980V02wLC6wFQRxDgphpGW')
    
    # Replace with your own search query
    query = f'("{word}" OR ({word} #{word}) OR "{word} price" OR #{word}trading) -is:retweet lang:en'

    tweets = tweet_client.search_recent_tweets(query=query, tweet_fields=['created_at'], max_results=count)
    
    # Creating list to append tweet data 
    tweets_list = []
    
    for tweet in tweets.data:
        
        #append the raw data into list
        tweets_list.append([clean_text(tweet.text),tweet.created_at])
        
    # Creating a dataframe from the tweets list above
    tweets_df1 = pd.DataFrame(tweets_list, columns=['Tweet', 'created_at'])

    tokenizer,model,labels=model_load()
    tweets_df1['sentiment'] = tweets_df1['Tweet'].apply(lambda x: compute_sentiment(tokenizer,model,labels,x))  
    
    return tweets_df1, tweets   


# """"#data to import in mongodb
# def import_data(tweets):
    
#     mydb = mongo_client["second"]
#     mycol = mydb["try"]
#    # Initialize an empty list
#     my_list = []

#     # Iterate over a range of values or a sequence
#     for i in tweets.data:
#             #msg= (i.created_at)
#         # Create a dictionary for each iteration
#         document = {"tweet": clean_text(i.text), "created_at": i.created_at}
#         # Append the dictionary to the list
#         my_list.append(document)

#     #insert data into mongodb client
#     try:
#         mycol.insert_many(my_list)
#     except:
#         pass  """

@st.experimental_memo

def draw_pie_chart(data):

    #count each sentiment
    sent = data['sentiment'].value_counts()
    
    #store the sentiment
    store = []
    for i in sent:
        store.append(i)
        
    #labels for annote
    mylabels = data['sentiment'].value_counts().index.tolist()
    
    if (len(mylabels)==2):
        # Creating explode data
        explode = (0.08, 0.0)
        # Creating color parameters
        colors = ( "#b36b00", "#ff6600")
        # Wedge properties
        wp = { 'linewidth' : 1 }
    elif (len(mylabels)==1):
        explode = (0.08)
        colors = ( "#b36b00")
        wp = { 'linewidth' : 1 }
    else:
       # mylabels = data['sentiment'].value_counts().index.tolist()
        explode = (0.08, 0.0, 0.0)
        colors = ( "#b36b00", "#ff6600", "#E3CF57")
        wp = { 'linewidth' : 1 }
        

    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%".format(pct, absolute)

    # Creating plot
    fig, ax = plt.subplots(figsize =(8,6 ))
    wedges, texts, autotexts = ax.pie(store,
                                      autopct = lambda pct: func(pct, store),
                                      explode = explode,
                                      labels = mylabels,
                                      shadow = False,
                                      colors = colors,
                                      startangle = 90,
                                      wedgeprops = wp,
                                      textprops = dict(color ="black"))
    # Adding legend
    ax.legend(wedges, mylabels,title ="Sentiments",loc ="center left",bbox_to_anchor =(1.2, 0, 0.5, 1.5))

    plt.setp(autotexts, size = 12, weight ="bold")
    return fig

def remove_stopwords(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Get the list of stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out the stopwords from the tokens
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text

def word_freq(df):
    # Remove stopwords from the 'tweets' column
    df['processed_tweet'] = df['Tweet'].apply(lambda word: remove_stopwords(word))
    new_df = df.processed_tweet.str.split(expand=True).stack().value_counts().reset_index()

    new_df.columns = ['Word', 'Frequency'] 
    
    fig= plt.figure(figsize=(10,8))
    
    sns.barplot(x='Frequency', y='Word',data=new_df[0:15])
   # plt.ylim([0,30])
    plt.xticks(rotation = 10, fontsize = 10)
    plt.yticks(fontsize = 16)
    plt.xlabel('Frequency', fontsize = 10)
    plt.ylabel('Words', fontsize = 10)
    plt.title('Top Most frequent Words in the collected Tweets', fontsize = 20)
    return fig 


def interpret_result(data,name):
    li = data['sentiment'].value_counts().index.tolist()
    
    if (len(li)==1):
        if (li[0]=='neutral'):
            msg= "The sentiment detected in all tweets is neutral, indicating no biased sentiment was found. As a result, it is not possible to make an absolute decision at this time."
        elif(len(li[0]=='Positive')):
            msg= f"The sentiment analysis result indicates a positive sentiment, suggesting that now would be a suitable time to consider purchasing the {name}."
        else:
            msg= f"The sentiment detected is negative, indicating that it is not advisable to invest in the {name} at this time."
    elif((len(li)==2) or (len(li)==3)):
        if((li[0]=='neutral') and (li[1]=='positive')):
            msg= f"The sentiment analysis results indicate that the majority of sentiments detected are neutral, followed by positive sentiment as the second most prevalent. Therefore, the current situation can be regarded as a moderately favorable time to consider investing in {name}."
        elif((li[0]=='neutral') and (li[1]=='negative')):
            msg= f"The sentiment analysis results indicate that the most prevalent sentiment detected is neutral, while the second highest sentiment detected is negative. Based on this, it can be inferred that the current time may not be ideal in investing {name}."
        elif(li[0]=='positive'):
            msg= f"The sentiment analysis result indicates a positive sentiment, suggesting that now would be a suitable time to consider purchasing the {name}."
        elif(li[0]=='negative'):
            msg= f"The sentiment detected is negative, indicating that it is not advisable to invest in the {name} at this time."

    return msg


@st.experimental_memo
def plot(word,count,name):
    df, tweets = data_extract(word,count)
    result = interpret_result(df,name)
    pie_chrt =  draw_pie_chart(df)
    #import_data(tweets)
    return result, pie_chrt,df

#************************************************************************


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: green;marginTop: -75px'>Sentiment Analysis of Social Media data</h1>", unsafe_allow_html=True)
cap = "Exploring the Impact of Social Media Data on Crypto Currency Price through Tweet Sentiment Analysis."

st.markdown(f'<p style="color:#402e70;font-weight: bold;font-size:18px;border-radius:2%;font-style: italic;">{cap}</p>', unsafe_allow_html=True)
#st.write("hi")
col=st.columns(2)
with col[0]:
        st.markdown("<h4 style= 'color: #fc036b;'>Select Your Crypto Currency </h4>", unsafe_allow_html=True)
        crypto_list = ["Bitcoin", "Ethereum", "Binance Coin", "Cardano", "XRP", "Dogecoin", "Polkadot", "Chainlink", "Litecoin", "Bitcoin Cash", "Stellar", "Uniswap", "Solana", "USD Coin", "VeChain"]
        crypto_name = st.selectbox("Enter details here ?", (crypto_list))

with col[1]:
        st.markdown("<h4 style= 'color: #fc036b;'>How many tweets(10-100)?</h4>", unsafe_allow_html=True)
        count=st.number_input("Enter details here",min_value=10, max_value=100)  


#button = st.button("Submit")
button = st.button("Submit") 

if(button):

    #import_data(tweets)
    if(len(crypto_name)>0):
       msg= ("Your Data is under Process. Please wait...")
       res1, res2, df = plot(crypto_name.lower(),count, crypto_name)
       col1=st.columns(3)

       with col1[0]:
            st.markdown(f'<p style="color:green;font-size:25px;border-radius:2%;">{res1}</p>', unsafe_allow_html=True)
            st.dataframe(df)

       with  col1[1]:
           st.write(res2)

       with col1[2]:
           viz = word_freq(df)
           st.write(viz)

       if st.button("Clear All"):
    # Clear values from *all* memoized functions:
    # i.e. clear values from both square and cube
          st.experimental_memo.clear()
    else:
        st.write("Incomplete Data")




