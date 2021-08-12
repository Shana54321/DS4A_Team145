# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:49:54 2021

@author: adeyi
"""

# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
os.environ['TOKEN'] ='AAAAAAAAAAAAAAAAAAAAAB3ZQQEAAAAA4nub9GI3gN1x%2B5%2FaW0bDbFqT63Y%3DLC30v9GAumiziWSkUPT53rQaHy24kV1LLFxiiZqT8MlmmObaqK'
def auth():
    return os.getenv('TOKEN')
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers
def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

# #Inputs for the request
# bearer_token = auth()
# headers = create_headers(bearer_token)
# keyword = "xbox lang:en"
# start_time = "2021-03-01T00:00:00.000Z"
# end_time = "2021-03-31T00:00:00.000Z"
# max_results = 15

# url = create_url(keyword, start_time,end_time, max_results)
# json_response = connect_to_endpoint(url[0], headers, url[1])
# #print the response in a readable format using this JSON library functions
# print(json.dumps(json_response, indent=4, sort_keys=True))
# json_response['data'][0]['created_at']
# #To retrieve the next_token for example, you can write:
# json_response['meta']['result_count']
# #To save results in JSON, we can easily do it using these two lines of code:
# with open('data.json', 'w') as f:
#     json.dump(json_response, f)
# #df = pd.DataFrame(response['json_response'])
# #df.to_csv('data.csv')

# #json_csv_approach
# # Create file
#csvFile = open("data.csv", "a", newline="", encoding='utf-8')
#csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save, in this example, we only want save these columns in our dataset
#csvWriter.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','tweet'])
#csvFile.close()
def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Geolocation
        # if ('geo' in tweet):   
        #     geo = tweet['geo']['place_id']
        # else:
        #     geo = " "

        # 4. Tweet ID
        tweet_id = tweet['id']

        # 5. Language
        lang = tweet['lang']

        # 6. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        # 7. source
        source = tweet['source']

        # 8. Tweet text
        text = tweet['text']
        
        # Assemble all data in a list
        res = [author_id, created_at, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter) 
#to loop
#Inputs for tweets
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "wendys lang:en"
start_list =    ['2016-01-01T00:00:00.000Z',
                 '2016-02-01T00:00:00.000Z',
                 '2016-03-01T00:00:00.000Z',
                 '2016-04-01T00:00:00.000Z',
                 '2016-05-01T00:00:00.000Z',
                 '2016-06-01T00:00:00.000Z',
                 '2016-07-01T00:00:00.000Z',
                 '2016-08-01T00:00:00.000Z',
                 '2016-09-01T00:00:00.000Z',
                 '2016-10-01T00:00:00.000Z',
                 '2016-11-01T00:00:00.000Z',
                 '2016-12-01T00:00:00.000Z',
                 '2017-01-01T00:00:00.000Z',
                 '2017-02-01T00:00:00.000Z',
                 '2017-03-01T00:00:00.000Z',
                 '2017-04-01T00:00:00.000Z',
                 '2017-05-01T00:00:00.000Z',
                 '2017-06-01T00:00:00.000Z',
                 '2017-07-01T00:00:00.000Z',
                 '2017-08-01T00:00:00.000Z',
                 '2017-09-01T00:00:00.000Z',
                 '2017-10-01T00:00:00.000Z',
                 '2017-11-01T00:00:00.000Z',
                 '2017-12-01T00:00:00.000Z',
                 '2018-01-01T00:00:00.000Z',
                 '2018-02-01T00:00:00.000Z',
                 '2018-03-01T00:00:00.000Z',
                 '2018-04-01T00:00:00.000Z',
                 '2018-05-01T00:00:00.000Z',
                 '2018-06-01T00:00:00.000Z',
                 '2018-07-01T00:00:00.000Z',
                 '2018-08-01T00:00:00.000Z',
                 '2018-09-01T00:00:00.000Z',
                 '2018-10-01T00:00:00.000Z',
                 '2018-11-01T00:00:00.000Z',
                 '2018-12-01T00:00:00.000Z',
                 '2019-01-01T00:00:00.000Z',
                 '2019-02-01T00:00:00.000Z',
                 '2019-03-01T00:00:00.000Z',
                 '2019-04-01T00:00:00.000Z',
                 '2019-05-01T00:00:00.000Z',
                 '2019-06-01T00:00:00.000Z',
                 '2019-07-01T00:00:00.000Z',
                 '2019-08-01T00:00:00.000Z',
                 '2019-09-01T00:00:00.000Z',
                 '2019-10-01T00:00:00.000Z',
                 '2019-11-01T00:00:00.000Z',
                 '2019-12-01T00:00:00.000Z',
                 '2020-01-01T00:00:00.000Z',
                 '2020-02-01T00:00:00.000Z',
                 '2020-03-01T00:00:00.000Z',
                 '2020-04-01T00:00:00.000Z',
                 '2020-05-01T00:00:00.000Z',
                 '2020-06-01T00:00:00.000Z',
                 '2020-07-01T00:00:00.000Z',
                 '2020-08-01T00:00:00.000Z',
                 '2020-09-01T00:00:00.000Z',
                 '2020-10-01T00:00:00.000Z',
                 '2020-11-01T00:00:00.000Z',
                 '2020-12-01T00:00:00.000Z',
                 '2021-01-01T00:00:00.000Z',
                 '2021-02-01T00:00:00.000Z',
                 '2021-03-01T00:00:00.000Z',
                 '2021-04-01T00:00:00.000Z',
                 '2021-05-01T00:00:00.000Z']

end_list =      ['2016-01-31T00:00:00.000Z',
                 '2016-02-28T00:00:00.000Z',
                 '2016-03-31T00:00:00.000Z',
                 '2016-04-30T00:00:00.000Z',
                 '2016-05-31T00:00:00.000Z',
                 '2016-06-30T00:00:00.000Z',
                 '2016-07-31T00:00:00.000Z',
                 '2016-08-31T00:00:00.000Z',
                 '2016-09-30T00:00:00.000Z',
                 '2016-10-31T00:00:00.000Z',
                 '2016-11-30T00:00:00.000Z',
                 '2016-12-31T00:00:00.000Z',
                 '2017-01-31T00:00:00.000Z',
                 '2017-02-28T00:00:00.000Z',
                 '2017-03-31T00:00:00.000Z',
                 '2017-04-30T00:00:00.000Z',
                 '2017-05-31T00:00:00.000Z',
                 '2017-06-30T00:00:00.000Z',
                 '2017-07-31T00:00:00.000Z',
                 '2017-08-31T00:00:00.000Z',
                 '2017-09-30T00:00:00.000Z',
                 '2017-10-31T00:00:00.000Z',
                 '2018-11-30T00:00:00.000Z',
                 '2018-12-31T00:00:00.000Z',
                 '2018-01-31T00:00:00.000Z',
                 '2018-02-28T00:00:00.000Z',
                 '2018-03-31T00:00:00.000Z',
                 '2018-04-30T00:00:00.000Z',
                 '2018-05-31T00:00:00.000Z',
                 '2018-06-30T00:00:00.000Z',
                 '2018-07-31T00:00:00.000Z',
                 '2018-08-31T00:00:00.000Z',
                 '2018-09-30T00:00:00.000Z',
                 '2018-10-31T00:00:00.000Z',
                 '2018-11-30T00:00:00.000Z',
                 '2018-12-31T00:00:00.000Z',
                 '2019-01-31T00:00:00.000Z',
                 '2019-02-28T00:00:00.000Z',
                 '2019-03-31T00:00:00.000Z',
                 '2019-04-30T00:00:00.000Z',
                 '2019-05-31T00:00:00.000Z',
                 '2019-06-30T00:00:00.000Z',
                 '2019-07-31T00:00:00.000Z',
                 '2019-08-31T00:00:00.000Z',
                 '2019-09-30T00:00:00.000Z',
                 '2019-10-31T00:00:00.000Z',
                 '2019-11-30T00:00:00.000Z',
                 '2019-12-31T00:00:00.000Z',
                 '2020-01-31T00:00:00.000Z',
                 '2020-02-28T00:00:00.000Z',
                 '2020-03-31T00:00:00.000Z',
                 '2020-04-30T00:00:00.000Z',
                 '2020-05-31T00:00:00.000Z',
                 '2020-06-30T00:00:00.000Z',
                 '2020-07-31T00:00:00.000Z',
                 '2020-08-31T00:00:00.000Z',
                 '2020-09-30T00:00:00.000Z',
                 '2020-10-31T00:00:00.000Z',
                 '2020-11-30T00:00:00.000Z',
                 '2020-12-31T00:00:00.000Z',
                 '2021-01-31T00:00:00.000Z',
                 '2021-02-28T00:00:00.000Z',
                 '2021-03-31T00:00:00.000Z',
                 '2021-04-30T00:00:00.000Z',
                 '2021-05-31T00:00:00.000Z']
max_results = 500

#Total number of tweets we collected from the loop
total_tweets = 0

# Create file
csvFile = open("data.csv", "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save, in this example, we only want save these columns in our dataset
csvWriter.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','tweet'])
csvFile.close()

for i in range(0,len(start_list)):

    # Inputs
    count = 0 # Counting tweets per time period
    max_count = 100 # Max tweets per time period
    flag = True
    next_token = None
    
    # Check if flag is true
    while flag:
        # Check if max_count reached
        if count >= max_count:
            break
        print("-------------------")
        print("Token: ", next_token)
        url = create_url(keyword, start_list[i],end_list[i], max_results)
        json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
        result_count = json_response['meta']['result_count']

        if 'next_token' in json_response['meta']:
            # Save the token to use for next call
            next_token = json_response['meta']['next_token']
            print("Next Token: ", next_token)
            if result_count is not None and result_count > 0 and next_token is not None:
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)                
        # If no next token exists
        else:
            if result_count is not None and result_count > 0:
                print("-------------------")
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "data.csv")
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)
            
            #Since this is the final request, turn flag to false to move to the next time period.
            flag = False
            next_token = None
        time.sleep(5)
print("Total number of results: ", total_tweets)