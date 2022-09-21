import pandas as pd
from pytrends.request import TrendReq

my_intrests = ["france", "germany", "russia", "united_kingdom", "united_states"]
for intrest in my_intrests :
	pytrend = TrendReq()

	df = pytrend.trending_searches(pn=intrest)
	df.to_csv("../data_trends/google_trending_topics_"+intrest+".csv")
