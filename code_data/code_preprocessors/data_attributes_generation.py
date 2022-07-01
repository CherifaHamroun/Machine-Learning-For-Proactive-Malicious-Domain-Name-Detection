import pandas as pd
import numpy as np
import tld
import tldextract
from itertools import groupby
import re
import time
import wordsegment as ws
import nostril as ns
import validators
from collections import Counter
from math import log
import multiprocessing as mp
import os
import shutil
from trending_keywords_extractor import trends_extract

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
def maliciousDataPreprocessing(filenames):
	list_dgas = []
	list_phish_dirty = []
	list_phish = []
	list_dgas_dirty = []
	list_spams = []
	list_spams_dirty = []
	with open(filenames[0], "r") as fd:
	    list_phish_dirty = fd.read().splitlines()
	for phish in list_phish_dirty:
	    if validators.domain(phish):
	    	list_phish.append(phish)
	with open(filenames[1], "r") as fd:
	    list_dgas_dirty = fd.read().splitlines()
	for dga in list_dgas_dirty:
	    domain = dga.split()[0]
	    if validators.domain(domain):
	        list_dgas.append(domain)
	with open(filenames[2], "r") as fd:
	    list_spams_dirty = fd.read().splitlines()
	for spam in list_spams_dirty:
	    if validators.domain(spam):
	        list_spams.append(spam)
	return (list_phish, list_dgas, list_spams)
def domainProcessor(domain):
        ws.load()
        nb_suspect_keywords = 0
        nb_legitimate_keywords = 0
        v_ctr = 0
        c_ctr = 0
        global list_lengths
        list_lengths.append(len(domain))
        txt = tldextract.extract(domain).suffix
        list_tld = txt.split(".")
        tld = list_tld[len(list_tld)-1]
        dom = domain[:-(len(tld)+1)]
        global list_tlds
        list_tlds.append(tld)
        global list_entropies
        list_entropies.append(shannon_entropy(domain))
        global list_positions
        list_positions.append(domain.rfind('-')/len(domain))
        count = 0
        last_x = ""
        freq_transition_d_c = 0
        for k, g in groupby(domain) :
            nb = len(list(g))
            if (nb>1) :
                count += nb
        global list_nb_consec_chars
        list_nb_consec_chars.append(count)
        list_chars = re.findall('[²&éè_çà)=ù@^`#~&$АаБбВвГгДдЄєЖжЅѕꙂꙃЗзꙀꙁИиІіЇїКкЛлМмНнОоПпРрСсТтѸѹУуФфХхѠѡЦцЧчШшЩщЪъЫыЬьѢѣЮюꙖꙗѤѥѦѧЯяѪѫѨѩѬѭѮѯѰѱѲѳѴѵõɛ̃ûυ\-]*',domain)
        global list_strange_chars
        list_strange_chars.append(len(list_chars)- list_chars.count(''))
        for word in suspect_keywords_list :
            nb_suspect_keywords += dom.count(word)
        for word in legitimate_keywords_list :
            nb_legitimate_keywords += dom.count(word)
        for x in domain:
            if (last_x.isdigit() and not x.isdigit()) or (not last_x.isdigit() and x.isdigit()):
                freq_transition_d_c +=1
            if x in vowels:
                v_ctr += 1
            elif x != '-' and not x.isdigit():
                c_ctr += 1
            last_x = x
        try:
            list_ratio_c_v.append(c_ctr/v_ctr)
        except ZeroDivisionError:
            list_ratio_c_v.append(70.0)
        global list_nb_suspect
        list_nb_suspect.append(nb_suspect_keywords)
        global list_nb_legitimate
        list_nb_legitimate.append(nb_legitimate_keywords)
        cpt = 0
        for trend in google_trends:
            for topic in trend:
                cpt = cpt+dom.count(topic)
        global list_trending_in_google_rate
        list_trending_in_google_rate.append(cpt)
        cpt = 0
        for trend in twitter_trends:
            for topic in trend:
                cpt = cpt+dom.count(topic)
        global list_trending_in_twitter_rate
        list_trending_in_twitter_rate.append(cpt)
        global list_nb_words
        list_nb_words.append(len(ws.segment(dom)))
        global list_nb_digits
        list_nb_digits.append(sum(c.isdigit() for c in domain))
        global list_freq_transitions
        list_freq_transitions.append(freq_transition_d_c)
        try:
            r = ns.nonsense(domain)
        except ValueError:
            r = True
        global list_nostril
        list_nostril.append(r)
        global list_cardinality
        list_cardinality.append(len(set(domain)))
        global list_nb_hyphends
        list_nb_hyphends.append(domain.count('-'))
        return{"len": list_lengths, 
            "tld": list_tlds, 
            "entropy": list_entropies, 
            "suspect" : list_nb_suspect, 
            "legitimate" : list_nb_legitimate,
            "consec": list_nb_consec_chars, 
            "strange" : list_strange_chars,
            "words":list_nb_words, 
            "digits":list_nb_digits, 
            "nonesense":list_nostril, 
            "cardinality":list_cardinality,
            "hyphends" : list_nb_hyphends,
            "position" : list_positions,
            "transitions" : list_freq_transitions,
            "ratio" : list_ratio_c_v,
            "trending_in_google":list_trending_in_google_rate,
            "trending_in_twitter":list_trending_in_twitter_rate,
           }

def lexicalDataGenerator(df) :
    ws.load()
    domains = df['domain_name']
    list_tlds = []
    list_entropies = []
    list_nb_consec_chars = []
    list_strange_chars = []
    list_lengths = []
    list_nb_suspect = []
    list_nb_legitimate = []
    list_nb_words = []
    list_nb_digits = []
    list_nostril = []
    list_nb_hyphends = []
    list_positions = []
    r = False
    list_cardinality = []
    list_ratio_c_v = []
    list_freq_transitions = []
    list_trending_in_google_rate=[]
    list_trending_in_twitter_rate=[]
    for domain in domains :
        nb_suspect_keywords = 0
        nb_legitimate_keywords = 0
        v_ctr = 0
        c_ctr = 0
        list_lengths.append(len(domain))
        txt = tldextract.extract(domain).suffix
        list_tld = txt.split(".")
        tld = list_tld[len(list_tld)-1]
        dom = domain[:-(len(tld)+1)]
        list_tlds.append(tld)
        list_entropies.append(shannon_entropy(domain))
        list_positions.append(domain.rfind('-')/len(domain))
        count = 0
        last_x = ""
        freq_transition_d_c = 0
        for k, g in groupby(domain) :
            nb = len(list(g))
            if (nb>1) :
                count += nb
        list_nb_consec_chars.append(count)
        list_chars = re.findall('[²&éè_çà)=ù@^`#~&$АаБбВвГгДдЄєЖжЅѕꙂꙃЗзꙀꙁИиІіЇїКкЛлМмНнОоПпРрСсТтѸѹУуФфХхѠѡЦцЧчШшЩщЪъЫыЬьѢѣЮюꙖꙗѤѥѦѧЯяѪѫѨѩѬѭѮѯѰѱѲѳѴѵõɛ̃ûυ\-]*',domain)
        list_strange_chars.append(len(list_chars)- list_chars.count(''))
        for word in suspect_keywords_list :
            nb_suspect_keywords += dom.count(word)
        for word in legitimate_keywords_list :
            nb_legitimate_keywords += dom.count(word)
        for x in domain:
            if (last_x.isdigit() and not x.isdigit()) or (not last_x.isdigit() and x.isdigit()):
                freq_transition_d_c +=1
            if x in vowels:
                v_ctr += 1
            elif x != '-' and not x.isdigit():
                c_ctr += 1
            last_x = x
        try:
            list_ratio_c_v.append(c_ctr/v_ctr)
        except ZeroDivisionError:
            list_ratio_c_v.append(70.0)
        list_nb_suspect.append(nb_suspect_keywords)
        list_nb_legitimate.append(nb_legitimate_keywords)
        cpt = 0
        for trend in google_trends:
            for topic in trend:
                cpt = cpt+dom.count(topic)
        list_trending_in_google_rate.append(cpt)
        cpt = 0
        for trend in twitter_trends:
            for topic in trend:
                cpt = cpt+dom.count(topic)
        list_trending_in_twitter_rate.append(cpt)
        list_nb_words.append(len(ws.segment(dom)))
        list_nb_digits.append(sum(c.isdigit() for c in domain))
        list_freq_transitions.append(freq_transition_d_c)
        try:
            r = ns.nonsense(domain)
        except ValueError:
            r = True
        list_nostril.append(r)
        list_cardinality.append(len(set(domain)))
        list_nb_hyphends.append(domain.count('-'))
    return{"len": list_lengths, 
            "tld": list_tlds, 
            "entropy": list_entropies, 
            "suspect" : list_nb_suspect, 
            "legitimate" : list_nb_legitimate,
            "consec": list_nb_consec_chars, 
            "strange" : list_strange_chars,
            "words":list_nb_words, 
            "digits":list_nb_digits, 
            "nonesense":list_nostril, 
            "cardinality":list_cardinality,
            "hyphends" : list_nb_hyphends,
            "position" : list_positions,
            "transitions" : list_freq_transitions,
            "ratio" : list_ratio_c_v,
            "trending_in_google":list_trending_in_google_rate,
            "trending_in_twitter":list_trending_in_twitter_rate,
           }
def dataProcessing(LARGE_FILE, CHUNKSIZE, cores):
	start_time = time.time()
	reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE)
	pool = mp.Pool(cores)
	result = {"len": [], "tld": [], "entropy": [], "suspect":[], "consec":[] , "strange" : [], "words" : [], "digits":[], "nonesense":[], "cardinality":[], "legitimate":[], "hyphends":[], "position":[], "transitions":[],"ratio":[], "trending_in_google":[], "trending_in_twitter":[]}
	for df in reader:
		f = pool.apply(lexicalDataGenerator,[df])
		result['len'] += f['len']
		result['tld'] += f['tld']
		result['entropy'] += f['entropy']
		result['suspect'] += f['suspect']
		result['consec'] += f['consec']
		result['strange'] += f['strange']
		result['words'] += f['words']
		result['digits'] += f['digits']
		result['nonesense'] += f['nonesense']
		result['cardinality'] += f['cardinality']
		result['legitimate'] += f['legitimate']
		result['hyphends'] += f['hyphends']
		result['position'] += f['position']
		result['transitions'] += f['transitions']
		result['ratio'] += f['ratio']
		result['trending_in_google'] += f['trending_in_google']
		result['trending_in_twitter'] += f['trending_in_twitter']
	print("--- %s seconds ---" % (time.time() - start_time))
	return result
if __name__ == '__main__':
	google_trends, twitter_trends = trends_extract()
	print('--- Domains list generation ... ')
	start_time = time.time()
	list_phish, list_dgas, list_spams = maliciousDataPreprocessing(["../data_sources/all_phishing.txt", "../data_sources/all_dgas.txt", "../data_sources/all_spams.txt"])
	list_gandi_selled_white = list(pd.read_csv('../data_sources/gandi_top_500k.csv')['Name'])
	list_alexa_white = list(pd.read_csv('../data_sources/alexa_top_1m.csv')['Name'])
	list_gandi_non_value_white = list(pd.read_csv('../data_sources/gandi_top_1m.csv')['product_detail'])
	domains = list_phish + list_dgas + list_spams + list_gandi_selled_white + list_alexa_white + list_gandi_non_value_white
	if os.path.exists("../data_all_lexical_features"):
		shutil.rmtree("../data_all_lexical_features")
	os.mkdir('../data_all_lexical_features')
	with open('../data_all_lexical_features/valid_domains_lists_length.txt', 'w') as f:
		print(len(domains),  file=f)
		print(len(list_phish), file=f)
		print(len(list_dgas), file=f)
		print(len(list_spams), file=f)
	classes = [1] * (len(list_phish) +len(list_dgas)+ len(list_spams)) + [0]*(len(list_alexa_white)+len(list_gandi_selled_white)+len(list_gandi_non_value_white))
	types = ['phish']*len(list_phish)+['dga']*len(list_dgas)+['spam']*len(list_spams)+['gandi_selled']*len(list_gandi_selled_white)+['alexa']*len(list_alexa_white)+['gandi_non_value']*len(list_gandi_non_value_white)
	df = pd.DataFrame(domains, columns = ['domain_name'])
	df['class'] = classes
	df.to_csv('../data_all_lexical_features/all_domains.csv')
	print("--- Data generated in  %s seconds ---" % (time.time() - start_time))
	LARGE_FILE = "../data_all_lexical_features/all_domains.csv"
	CHUNKSIZE = int(len(domains)/os.cpu_count())+1
	cores = os.cpu_count()
	start_time = time.time()
	print('--- Data processing ... ')
	result = dataProcessing(LARGE_FILE, CHUNKSIZE, cores)
	print("--- Data processed in  %s seconds ---" % (time.time() - start_time))
	start_time = time.time()
	print('--- Data stacking ... ')
	df = pd.DataFrame(domains, columns = ['domain_name'])
	df['domain_len'] = result['len']
	df['tld'] = result['tld']
	df['entropy'] = result['entropy']
	df['nb_suspect_keywords'] = result['suspect']
	df['nb_consec_chars'] = result['consec']
	df['nb_strange_chars'] = result['strange']
	df['nb_words'] = result['words']
	df['nb_digits'] = result['digits']
	df['nonesense'] = result['nonesense']
	df['cardinality'] = result['cardinality']
	df['nb_legitimate_keywords'] = result['legitimate']
	df['nb_hyphends'] = result['hyphends']
	df['last_hyphend_position'] = result['position']
	df['transition_frequency'] = result['transitions']
	df['consonent_vowel_ratio'] = result['ratio']
	df['google_trend_rate'] = result['trending_in_google']
	df['twitter_trend_rate'] = result['trending_in_twitter']
	df['class'] = classes
	df['type'] = types
	df.to_csv('../data_all_lexical_features/all_domains_with_textual_attributes.csv')
	print("--- Data stacked in  %s seconds ---" % (time.time() - start_time))
