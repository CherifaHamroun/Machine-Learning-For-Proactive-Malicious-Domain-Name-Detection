import tweepy 
import config
from twitter import *
import os

auth = tweepy.OAuthHandler(config.api_key, config.api_key_secret) 
auth.set_access_token(config.access_token, config.access_token_secret) 
api = tweepy.API(auth) 
twitter = Twitter(auth = OAuth(config.access_token,
                  config.access_token_secret,
                  config.api_key,
                  config.api_key_secret))
my_intrests = ["Worldwide", "France", "Germany", "Russia", "United Kingdom","United States"]
for available in api.available_trends():
	if available["name"] in my_intrests :
		results = twitter.trends.place(_id = available["woeid"])
		for location in results:
			for trend in location["trends"]:
				with open("../data_trends/twitter_trending_topics_"+available["name"]+".txt", "a") as text_file:
					text_file.write("%s \n" % trend['name'])
