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
from elm_pure import ELM

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
	
	df = pd.read_csv('../../code_data/all_domains_with_textual_attributes.csv')
	elm_from_joblib = joblib.load('./models_overall/elm_spams_dgas_phishing_gandi_selled.pkl')
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
	tlds = new_le.fit_transform(np.append(df['tld'].to_numpy(),tld))
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
	attributes = da.reshape(da.from_array([len(domain),tlds[tlds.shape[0]-1],shannon_entropy(domain),nb_suspect_keywords,count,len(list_chars)- list_chars.count(''), len(ws.segment(domain)),sum(c.isdigit() for c in domain),int(r),len(set(domain)), nb_legitimate_keywords, domain.count('-'), domain.find('-')/len(domain) ,freq_transition_d_c ,ratio]),(1,15))
	prediction = elm_from_joblib.predict(attributes)
	print(np.round(prediction))
