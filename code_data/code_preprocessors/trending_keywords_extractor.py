import pandas as pd
import os
import numpy as np
def trends_extract():

	list_temp1_trends=[]
	list_temp2_trends=[]
	list_google_trends=[]
	list_twitter_trends=[]
	for filename in os.listdir('../data_trends'):
		f = os.path.join('../data_trends', filename)
		if os.path.isfile(f) and filename[-4:]=='.csv':
			list_temp1_trends = list_temp1_trends + list(pd.read_csv(f)['0'])
		else:
			with open(f, "r") as fd:
				list_temp2_trends = fd.read().splitlines()
	for trend in list_temp1_trends:
		treated_trend = trend.lower().split()
		if treated_trend not in list_google_trends : 
			list_google_trends.append(treated_trend)
	for trend in list_temp2_trends:
		treated_trend = trend.lower().split()
		if treated_trend not in list_twitter_trends :
			list_twitter_trends.append(treated_trend)
	return(list_google_trends, list_twitter_trends)
if __name__=='__main__':
	print('google ',len(trends_extract()[0]))
	print('twitter',len(trends_extract()[1]))
