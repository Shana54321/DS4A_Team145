# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 09:39:46 2021

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import seaborn
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

df = pd.read_csv('GENERALMOTORS_final.csv')
df.head()
df_new=df.drop(['vix_adj_close', 'adj _close','date','id'], axis = 1)

  
############# Main Section ############
# loading dataset using seaborn
# pairplot with hue day
seaborn.pairplot(df_new)
# to show
plt.show()
#missing value
df_new.info()
df_new=df_new.fillna(0)
#df_new.head()
df_new.info()
df_new.rename(columns = {'close_previous':'Future_close'}, inplace = True)
# Python3 code to show Box-cox Transformation 
# of non-normal data
  
# import modules
import numpy as np
from scipy import stats

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

df_new['neutral'].abs()
jj=['VolStat','negative','positive','neutral','volume'] 
for i in jj:    
# transform training data & save lambda value
    fitted_data, fitted_lambda = stats.boxcox(df_new['VolStat'])
      
    # creating axes to draw plots
    fig, ax = plt.subplots(1, 2)
      
    # plotting the original data(non-normal) and 
    # fitted data (normal)
    sns.distplot(df_new['VolStat'], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Non-Normal", color ="green", ax = ax[0])
      
    sns.distplot(fitted_data, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Normal", color ="green", ax = ax[1])
      
    # adding legends to the subplots
    plt.legend(loc = "upper right")
      
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
    df_new['VolStat']=fitted_data
    print(f"Lambda value used for Transformation: {fitted_lambda}")
 #negative   
    # transform training data & save lambda value
    fitted_data_1, fitted_lambda_1 = stats.boxcox(df_new['negative'])
      
    # creating axes to draw plots
    fig, ax = plt.subplots(1, 2)
      
    # plotting the original data(non-normal) and 
    # fitted data (normal)
    sns.distplot(df_new['negative'], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Non-Normal", color ="green", ax = ax[0])
      
    sns.distplot(fitted_data_1, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Normal", color ="green", ax = ax[1])
      
    # adding legends to the subplots
    plt.legend(loc = "upper right")
      
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
    df_new['negative']=fitted_data_1
    print(f"Lambda value used for Transformation: {fitted_lambda_1}")
 #positive    
    # transform training data & save lambda value
    fitted_data_2, fitted_lambda_2 = stats.boxcox(df_new['positive'])
      
    # creating axes to draw plots
    fig, ax = plt.subplots(1, 2)
      
    # plotting the original data(non-normal) and 
    # fitted data (normal)
    sns.distplot(df_new['positive'], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Non-Normal", color ="green", ax = ax[0])
      
    sns.distplot(fitted_data_2, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Normal", color ="green", ax = ax[1])
      
    # adding legends to the subplots
    plt.legend(loc = "upper right")
      
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
    df_new['positive']=fitted_data_2
    print(f"Lambda value used for Transformation: {fitted_lambda_2}")
#neutral    
    # transform training data & save lambda value
    fitted_data_3, fitted_lambda_3 = stats.boxcox(df_new['neutral'])
      
    # creating axes to draw plots
    fig, ax = plt.subplots(1, 2)
      
    # plotting the original data(non-normal) and 
    # fitted data (normal)
    sns.distplot(df_new['neutral'], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Non-Normal", color ="green", ax = ax[0])
      
    sns.distplot(fitted_data_3, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Normal", color ="green", ax = ax[1])
      
    # adding legends to the subplots
    plt.legend(loc = "upper right")
      
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
    df_new['neutral']=fitted_data_3
    print(f"Lambda value used for Transformation: {fitted_lambda_3}")
#volume  
    # transform training data & save lambda value
    fitted_data_4, fitted_lambda_4 = stats.boxcox(df_new['volume'])
      
    # creating axes to draw plots
    fig, ax = plt.subplots(1, 2)
      
    # plotting the original data(non-normal) and 
    # fitted data (normal)
    sns.distplot(df_new['volume'], hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Non-Normal", color ="green", ax = ax[0])
      
    sns.distplot(fitted_data_4, hist = False, kde = True,
                kde_kws = {'shade': True, 'linewidth': 2}, 
                label = "Normal", color ="green", ax = ax[1])
      
    # adding legends to the subplots
    plt.legend(loc = "upper right")
      
    # rescaling the subplots
    fig.set_figheight(5)
    fig.set_figwidth(10)
    df_new['volume']=fitted_data_4
    print(f"Lambda value used for Transformation: {fitted_lambda_4}")  
#data with negative value.
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
pt.fit(df_new['Future_close'].values.reshape(-1,1))
df_new['Future_close'] = pt.transform(df_new['Future_close'].values.reshape(-1,1))
plt.ylabel('Count')
plt.xlabel('Box Cox Transformed Negative Sentiment')
plt.title('Dist. of Transformed Negative Sentiment - Feature')
plt.hist(df_new['Future_close'])

pt = PowerTransformer()
pt.fit(df_new['negative'].values.reshape(-1,1))
df_new['negative'] = pt.transform(df_new['negative'].values.reshape(-1,1))
plt.ylabel('Count')
plt.xlabel('Box Cox Transformed Negative Sentiment')
plt.title('Dist. of Transformed Negative Sentiment - Feature')
plt.hist(df_new['negative'])pt = PowerTransformer()
pt.fit(df_new['neutral'].values.reshape(-1,1))
df_new['neutral'] = pt.transform(df_new['neutral'].values.reshape(-1,1))
plt.ylabel('Count')
plt.xlabel('Box Cox Transformed Negative Sentiment')
plt.title('Dist. of Transformed Negative Sentiment - Feature')
plt.hist(df_new['neutral'])

pt = PowerTransformer()
pt.fit(df_new['positive'].values.reshape(-1,1))
df_new['positive'] = pt.transform(df_new['positive'].values.reshape(-1,1))
plt.ylabel('Count')
plt.xlabel('Box Cox Transformed Negative Sentiment')
plt.title('Dist. of Transformed Negative Sentiment - Feature')
plt.hist(df_new['positive'])





# We specify random seed so that the train and test data set does not have the same rows, respectively
#np.random.seed(0)
#df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train, df_test = train_test_split(df_new, train_size = 0.7, test_size = 0.3,shuffle=False)

scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['Future_close', 'VolStat', 'negative', 'neutral', 'positive','volume']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

new_df_train=df_train.copy()
# Dividing the training data set into X and Y
y_train = df_train.pop('Future_close')
X_train = df_train
#Build a linear model
X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
lr_1.summary()


X=X_train.drop('volume',1)
#Build a linear model
X_train_lm = sm.add_constant(X)
lr_2 = sm.OLS(y_train, X_train_lm).fit()
lr_2.summary()

X=X.drop('VolStat',1)
#Build a linear model
X_train_lm = sm.add_constant(X)
lr_3 = sm.OLS(y_train, X_train_lm).fit()
lr_3.summary()


new_df_train['stocks_pred'] = lr_1.predict(X_train_lm)
new_df_train['residual'] = lr_1.resid
new_df_train.head()
#Multicolinearity
corr = new_df_train.corr()
print('Pearson correlation coefficient matrix of each variables:\n', corr)
# plotting correlation heatmap
dataplot =sns.heatmap(corr, cmap="YlGnBu", annot=True) 
dataplot =sns.heatmap(corr, cmap='RdYlGn')  
# displaying heatmap
plt.show()


# Plotting the observed vs predicted values
sns.lmplot(x='Future_close', y='stocks_pred', data=new_df_train, fit_reg=False, size=5)
    
# Plotting the diagonal line
line_coords = np.arange(new_df_train[['Future_close', 'stocks_pred']].min().min()-10, 
                        new_df_train[['Future_close', 'stocks_pred']].max().max()+10)
plt.plot(line_coords, line_coords,  # X and y points
         color='darkorange', linestyle='--')

plt.ylabel('Predicted Stocks', fontsize=14)
plt.xlabel('Actual Stocks', fontsize=14)
plt.title('Linearity Assumption', fontsize=16)
plt.show()



# Checking for the VIF values of the variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Creating a dataframe that will contain the names of all the feature variables and their VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables and insignificant variables
X = X_train.drop('negative', 1,)

# Build a fitted model after dropping the variable
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()

# Printing the summary of the model
print(lr_2.summary())
#plotting residuals
import seaborn as sns
y_train_price = lr_1.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label