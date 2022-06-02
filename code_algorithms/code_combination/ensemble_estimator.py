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
import sys
sys.path.insert(0, '../code_elmnn/')
from elm_pure import ELM
import pickle

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
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	df_phishing = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_phishing_alike.csv')
	df_spams = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_spams_alike.csv')
	df_dgas = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_dgas_alike.csv')
	df_alexa = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_alexa_alike.csv')
	df_gandi_selled = pd.read_csv('../../code_data/instances_for_alikeness_test/instance_gandi_selled_alike.csv')
	elm_from_joblib_alexa = joblib.load("../code_elmnn/models_alikeness/elm_alexa_alike.pkl")
	elm_from_joblib_gandi_selled = joblib.load("../code_elmnn/models_alikeness/elm_gandi_selled_alike.pkl")
	elm_from_joblib_phishing = joblib.load("../code_elmnn/models_alikeness/elm_phishing_alike.pkl")
	elm_from_joblib_spams = joblib.load("../code_elmnn/models_alikeness/elm_spams_alike.pkl")
	elm_from_joblib_dgas = joblib.load("../code_elmnn/models_alikeness/elm_dgas_alike.pkl")
	rf_from_pickle_alexa = pickle.load(open("../code_rf/models_alikeness/rf_alexa_alike.pkl", "rb"))
	rf_from_pickle_gandi_selled = pickle.load(open("../code_rf/models_alikeness/rf_gandi_selled_alike.pkl", "rb"))
	rf_from_pickle_phishing = pickle.load(open("../code_rf/models_alikeness/rf_phishing_alike.pkl", "rb"))
	rf_from_pickle_spams = pickle.load(open("../code_rf/models_alikeness/rf_spams_alike.pkl", "rb"))
	rf_from_pickle_dgas = pickle.load(open("../code_rf/models_alikeness/rf_dgas_alike.pkl", "rb"))
	xgb_from_pickle_alexa = pickle.load(open("../code_xgboost/models_alikeness/xgbdask_alexa_alike.pkl", "rb"))
	xgb_from_pickle_gandi_selled = pickle.load(open("../code_xgboost/models_alikeness/xgbdask_gandi_selled_alike.pkl", "rb"))
	xgb_from_pickle_phishing = pickle.load(open("../code_xgboost/models_alikeness/xgbdask_phishing_alike.pkl", "rb"))
	xgb_from_pickle_spams = pickle.load(open("../code_xgboost/models_alikeness/xgbdask_spams_alike.pkl", "rb"))
	xgb_from_pickle_dgas = pickle.load(open("../code_xgboost/models_alikeness/xgbdask_dgas_alike.pkl", "rb"))
	domain = input("Domain name : ")
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
		ratio = 500.0
	nb_legitimate_keywords = 0
	for word in legitimate_keywords_list :
		nb_legitimate_keywords += domain.count(word)
	attributes_phishing = da.reshape(da.from_array([len(domain),tlds_phishing[tlds_phishing.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))
	attributes_spams = da.reshape(da.from_array([len(domain),tlds_spams[tlds_spams.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))
	attributes_dgas = da.reshape(da.from_array([len(domain),tlds_dgas[tlds_dgas.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))
	attributes_alexa = da.reshape(da.from_array([len(domain),tlds_alexa[tlds_alexa.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))
	attributes_gandi_selled = da.reshape(da.from_array([len(domain),tlds_gandi_selled[tlds_gandi_selled.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))	
	phishing_rate = (np.round(rf_from_pickle_phishing.predict_proba(attributes_phishing)), xgb.dask.predict(client, xgb_from_pickle_phishing,attributes_phishing).compute()[0], elm_from_joblib_phishing.predict(attributes_phishing))
	spam_rate = ( np.round(rf_from_pickle_spams.predict_proba(attributes_spams)), xgb.dask.predict(client, xgb_from_pickle_spams,attributes_spams).compute()[0], elm_from_joblib_spams.predict(attributes_spams))
	dga_rate = ( np.round(rf_from_pickle_dgas.predict_proba(attributes_dgas)), xgb.dask.predict(client, xgb_from_pickle_dgas,attributes_dgas).compute()[0], elm_from_joblib_dgas.predict(attributes_dgas))
	gandi_selled_rate = (np.round(rf_from_pickle_gandi_selled.predict_proba(attributes_gandi_selled)), xgb.dask.predict(client, xgb_from_pickle_gandi_selled,attributes_gandi_selled).compute()[0], elm_from_joblib_gandi_selled.predict(attributes_gandi_selled))
	alexa_rate = (np.round(rf_from_pickle_alexa.predict_proba(attributes_alexa)), xgb.dask.predict(client, xgb_from_pickle_alexa,attributes_alexa).compute()[0], elm_from_joblib_alexa.predict(attributes_alexa))
	print(phishing_rate, spam_rate, dga_rate, gandi_selled_rate, alexa_rate)
	#maliciousness_rate = phishing_rate + spam_rate+ dga_rate
	#benign_rate = gandi_selled_rate + alexa_rate
	#print (maliciousness_rate, benign_rate)
