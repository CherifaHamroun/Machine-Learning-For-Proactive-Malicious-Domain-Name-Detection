import pandas as pd
from pytrends.request import TrendReq

my_intrests = ["france", "germany", "russia", "united_kingdom", "united_states"]
for intrest in my_intrests :
#pytrend = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})
	pytrend = TrendReq()
#print(pytrend.build_payload(kw_list=['phish']))
#print(pytrend.build_payload(kw_list=['spam']))
#print(pytrend.build_payload(kw_list=['typosquat']))

	df = pytrend.trending_searches(pn=intrest)
	df.to_csv("../data_trends/google_trending_topics_"+intrest+".csv")
