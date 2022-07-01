import joblib
import xgboost as xgb
import dask.distributed
import tldextract
import wordsegment as ws
from itertools import groupby
import nostril as ns
import re
from collections import Counter
from math import log
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np 
import dask.array as da
import pickle
import sys
sys.path.append('../../')
from code_data.code_preprocessors.trending_keywords_extractor import trends_extract
import os

suspect_keywords_list = ['activity', 'appleid', 'poloniex', 'moneygram', 'overstock', '.bank', '.online', '.comalert', 'online',
 'coinhive', '.business', '.party', '-com.', 'purchase', 'recover', 'iforgot', 'bithumb', 'reddit', '.cc', '.pw', '.netauthentication', 'safe',
 'kraken', 'wellsfargo', '.center', '.racing', '.orgauthorize', 'secure', 'protonmail' ,'localbitcoin' , 'ukraine' , '.cf', '.ren' 'cgi-bin',
 'bill', 'security', 'tutanota', 'bitstamp', '.click' , '.review', '.comclient', 'service', 'bittrex', 'santander', '.club', '.science', '.net.',
 'support', 'transaction', 'blockchain', 'flickr', 'morganstanley', '.country', '.stream', '.org.', 'unlock', 'update', 'bitflyer', 'barclays',
 '.download', '.study', '.com.', 'wallet', 'account', 'outlook', 'coinbase', 'hsbc', '.ga', '.support', '.govform', 'login', 'hitbtc', 'scottrade',
 '.gb', '.tech', '.gov.', 'log-in', 'password', 'lakebtc', 'ameritrade', '.gdn', '.tk' , '.gouvlive', 'signin', 'bitfinex', 'merilledge', '.gq' ,
 '.top' , '-gouvmanage', 'sign-in', 'bitconnect', 'bank', '.info', '.vip', '.gouv.', 'verification', 'verify', 'coinsbank', '.kim' ,
 '.win', 'webscr', 'invoice', '.loan', '.work', 'authenticate', 'confirm', '.men', '.xin', 'credential', 'customer', '.ml', '.xyz', '.mom', 'phish',
 'viagra','russia', 'billing', 'casino', 'spam']

legitimate_keywords_list = ['outlook', 'facebook', 'skype', 'icloud', 'office365', 'tumblr', 'westernunion', 'alibaba', 'github', 'microsoft', 'reddit',
 'bankofamerica', 'aliexpress', 'itunes', 'windows', 'youtube', 'leboncoin', 'apple', 'twitter' 'paypal', 'amazon', 'netflix', 'linkedin', 'citigroup',
 'hotmail', 'bittrex', 'instagram', 'gmail', 'google',  'whatsapp', 'yahoo' , 'yandex', 'baidu', 'bilibili', 'wikipedia', 'qq', 'zhihu', 'reddit', 'bing',
 'taobao', 'csdn', 'live', 'weibo', 'sina', 'zoom', 'office', 'sohu', 'tmail', 'tiktok', 'canva', 'stackoverflow', 'naver', 'fandom', 'mail', 'myshopify',
 'douban' ]

vowels = ['a', 'e', 'i', 'o', 'u']

def shannon_entropy(string):
    counts = Counter(string)
    frequencies = ((i / len(string)) for i in counts.values())
    return - sum(f * log(f, 2) for f in frequencies)

if __name__=='__main__':
	df_phishing = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_phish_alike.csv')
	df_spams = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_spam_alike.csv')
	df_dgas = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_dga_alike.csv')
	df_alexa = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_alexa_alike.csv')
	df_gandi_selled = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_gandi_selled_alike.csv')
	df_gandi_non_value = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_gandi_non_value_alike.csv')
	onesvm_from_pickle_alexa = pickle.load(open("./models_alikeness/if_alexa_alike.pkl", "rb"))
	onesvm_from_pickle_gandi_selled = pickle.load(open("./models_alikeness/if_gandi_selled_alike.pkl", "rb"))
	onesvm_from_pickle_phishing = pickle.load(open("./models_alikeness/if_phish_alike.pkl", "rb"))
	onesvm_from_pickle_spams = pickle.load(open("./models_alikeness/if_spam_alike.pkl", "rb"))
	onesvm_from_pickle_dgas = pickle.load(open("./models_alikeness/if_dga_alike.pkl", "rb"))
	onesvm_from_pickle_gandi_non_value = pickle.load(open("./models_alikeness/if_gandi_non_value_alike.pkl", "rb"))
	os.chdir("../../code_data/code_preprocessors")
	google_trends, twitter_trends = trends_extract()
	cpt_google = 0
	cpt_twitter = 0
	domain = sys.argv[1]
	for trend in google_trends:
		for topic in trend:
			cpt_google = cpt_google+domain.count(topic)
	for trend in twitter_trends:
		for topic in trend:
			cpt_twitter = cpt_twitter+domain.count(topic)
	ws.load()
	txt = tldextract.extract(domain).suffix
	list_tld = txt.split(".")
	tld = list_tld[len(list_tld)-1]
	count = 0
	for k, g in groupby(domain) :
		nb = len(list(g))
		if (nb>1) :
			count += nb
	nb_suspect_keywords= 0
	for word in suspect_keywords_list :
		nb_suspect_keywords += domain.count(word)
	list_chars = re.findall('[²&éè_çà)=ù@^`#~&$АаБбВвГгДдЄєЖжЅѕꙂꙃЗзꙀꙁИиІіЇїКкЛлМмНнОоПпРрСсТтѸѹУуФфХхѠѡЦцЧчШшЩщЪъЫыЬьѢѣЮюꙖꙗѤѥѦѧЯяѪѫѨѩѬѭѮѯѰѱѲѳѴѵõɛ̃ûυ\-]*',domain)
	try:
		r = ns.nonsense(domain)
	except ValueError:
		r = True
	new_le = LabelEncoder()
	tlds_phishing = new_le.fit_transform(np.append(df_phishing['tld'].to_numpy(),tld))
	tlds_spams = new_le.fit_transform(np.append(df_spams['tld'].to_numpy(),tld))
	tlds_dgas = new_le.fit_transform(np.append(df_dgas['tld'].to_numpy(),tld))
	tlds_alexa = new_le.fit_transform(np.append(df_alexa['tld'].to_numpy(),tld))
	tlds_gandi_selled = new_le.fit_transform(np.append(df_gandi_selled['tld'].to_numpy(),tld))
	freq_transition_d_c = 0
	x = ''
	last_x = ''
	c_ctr = 0
	v_ctr = 0
	for x in domain:
		if (last_x.isdigit() and not x.isdigit()) or (not last_x.isdigit() and x.isdigit()):
			freq_transition_d_c +=1
		if x in vowels:
			v_ctr += 1
		elif x != '-' and not x.isdigit():
			c_ctr += 1
		last_x = x
	try:
		ratio = c_ctr/v_ctr
	except ZeroDivisionError:
		ratio = 70.0
	nb_legitimate_keywords = 0
	for word in legitimate_keywords_list :
		nb_legitimate_keywords += domain.count(word)
	attributes = da.reshape(da.from_array([len(domain),tlds_phishing[tlds_phishing.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio, cpt_google, cpt_twitter]),(1,17))

	phishing_rate = onesvm_from_pickle_phishing.predict(attributes)
	spam_rate = onesvm_from_pickle_spams.predict(attributes)
	dga_rate = onesvm_from_pickle_dgas.predict(attributes)
	gandi_selled_rate = onesvm_from_pickle_gandi_selled.predict(attributes)
	alexa_rate = onesvm_from_pickle_alexa.predict(attributes)
	gandi_non_value_rate = onesvm_from_pickle_gandi_non_value.predict(attributes)
	malicious_rate = phishing_rate + spam_rate + spam_rate
	benign_rate = alexa_rate + gandi_non_value_rate + gandi_selled_rate
	if malicious_rate>benign_rate :
		prediction = 'True'
	elif malicious_rate<benign_rate :
		prediction = 'False'
	else : 
		prediction = 'Suspect'
	with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
		file1.write(prediction +'\n')
