# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:32:04 2021

@author: adeyi
"""
import pandas as pd
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import base64

df = pd.read_csv('data_k.csv')
# df["quotesymbol"]="K"
# # Create file
# df.to_csv('combined_csv.csv', index=False)
# import os
# import glob
# import pandas as pd
# os.chdir("data")
# extension = 'csv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
# #export to csv
# combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
# positive = 0
# negative = 0
# neutral = 0
# polarity = 0
tweet_list = []
# neutral_list = []
# negative_list = []
# positive_list = []
# for k in df.tweet: 
#  #print(tweet.text)
#     tweet_list.append(k)
#     analysis = TextBlob(k)
#     score= SentimentIntensityAnalyzer().polarity_scores(k)
#     neg = score["neg"]
#     neu = score["neu"]
#     pos = score["pos"]
#     comp = score["compound"]
#     polarity += analysis.sentiment.polarity 
#     if neg > pos:
#         negative_list.append(k)
#         negative += 1
#     elif pos > neg:
#         positive_list.append(k)
#         positive += 1
 
#     elif pos == neg:
#         neutral_list.append(k)
#         neutral += 1
        
tweet_list= df['tweet'].tolist()
# positive = percentage(positive, noOfTweet)
# negative = percentage(negative, noOfTweet)
# neutral = percentage(neutral, noOfTweet)
# polarity = percentage(polarity, noOfTweet)
# positive = format(positive, ".1f")
# negative = format(negative, ".1f")
# neutral = format(neutral, ".1f")
#tweet_list.head(10)
tw_list=pd.DataFrame(tweet_list)
#len(tweet_list)
#Cleaning Text (RT, Punctuation etc)
#Creating new dataframe and new features
tw_list['text'] = tw_list[0]
tw_list['text'] = tw_list['text'].apply(str) #converting the values in the column to string.
#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
# tw_list['text'] = tw_list['text'].apply(str) #converting the values in the column to string.
tw_list['text'] = tw_list.text.map(remove_rt).map(rt)
tw_list['text'] = tw_list.text.str.lower()
#tw_list.head(10)
#Calculating Negative, Positive, Neutral and Compound values
tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp
merged_df=pd.concat([df, tw_list], axis=1)
merged_df.drop_duplicates(inplace = True)
merged_df.to_csv('tw_merged_df.csv', index=False)
#tw_list.head(10)
#label
merged_df_1_labels=merged_df.copy()
def label(pp):
    from sklearn.preprocessing import LabelEncoder
    cols = ["sentiment"]
    lbl = LabelEncoder()
    pred_lbl=lbl.fit_transform(pp[cols])
    mappings = {index: label for index, label in enumerate(lbl.classes_)}
    pp["sentiment_label"]=pred_lbl
    return(pp)
mappings

merged_df_1_labels.to_csv('tw_merged_df_labels.csv', index=False)
merged_df_1_labels['created_at']=pd.to_datetime(merged_df_1_labels['created_at'])
daily_tw=merged_df_1_labels.groupby(merged_df_1_labels['created_at'].dt.to_period(freq='D'))['sentiment'].size()#Grouping by months
import datetime as dt
df3=df2[['author id','created_at','sentiment']]
df3['created_at']=pd.to_datetime(df3['created_at']).dt.date
df3.head(100000)
df4= df3.groupby(['created_at','sentiment']).nunique()
df4.head(1000000)
com_twt= df4.pivot_table(index=['created_at'],columns=['sentiment'],values='author id')

com_twt.head(10000)