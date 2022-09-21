import joblib
import xgboost as xgb
import dask.distributed
import dask.dataframe as ddf
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
import sys
sys.path.append('../')
from code_data.code_preprocessors.trending_keywords_extractor import trends_extract
import os
import shutil
import pickle
import joblib
from joblib import Parallel, delayed

def shannon_entropy(string):
    counts = Counter(string)
    frequencies = ((i / len(string)) for i in counts.values())
    return - sum(f * log(f, 2) for f in frequencies)

suspect_keywords_list = ['activity', 'appleid', 'poloniex', 'moneygram', 'overstock', '.bank', '.online', '.comalert', 'online',
 'coinhive', '.business', '.party','.com' ,'-com.', 'purchase', 'recover', 'iforgot', 'bithumb', 'reddit', '.cc', '.pw', '.netauthentication', 'safe',
 'kraken', 'wellsfargo', '.center', '.racing', '.orgauthorize', 'secure', 'protonmail' ,'localbitcoin' , 'ukraine' , '.cf', '.ren' 'cgi-bin',
 'bill', 'security', 'tutanota', 'bitstamp', '.click' , '.review', '.comclient', 'service', 'bittrex', 'santander', '.club', '.science', '.net.',
 'support', 'transaction', 'blockchain', 'flickr', 'morganstanley', '.country', '.stream', '.org.', 'unlock', 'update', 'bitflyer', 'barclays',
 '.download', '.study', '.com.', 'wallet', 'account', 'outlook', 'coinbase', 'hsbc', '.ga', '.support', '.govform', 'login', 'hitbtc', 'scottrade',
 '.gb', '.tech', '.gov.', 'log-in', 'password', 'lakebtc', 'ameritrade', '.gdn', '.tk' , '.gouvlive', 'signin', 'bitfinex', 'merilledge', '.gq' ,
 '.top' , '-gouvmanage', 'sign-in', 'bitconnect', 'bank', '.info', '.vip', '.gouv.', 'verification', 'verify', 'coinsbank', '.kim' ,
 '.win', 'webscr', 'invoice', '.loan', '.work', 'authenticate', 'confirm', '.men', '.xin', 'credential', 'customer', '.ml', '.xyz', '.mom', 'phish',
 'viagra','russia', 'billing', 'casino', 'spam', 'fuck', 'hate', 'drug', 'violence', 'viagra', 'scam', 'dga', 'typo', 'suck', 'payment', 'out', 'in', 'enter']


legitimate_keywords_list = ['outlook', 'facebook', 'skype', 'icloud', 'office365', 'tumblr', 'westernunion', 'alibaba', 'github', 'microsoft', 'reddit',
 'bankofamerica', 'aliexpress', 'itunes', 'windows', 'youtube', 'leboncoin', 'apple', 'twitter' 'paypal', 'amazon', 'netflix', 'linkedin', 'citigroup',
 'hotmail', 'bittrex', 'instagram', 'gmail', 'google',  'whatsapp', 'yahoo' , 'yandex', 'baidu', 'bilibili', 'wikipedia', 'qq', 'zhihu', 'reddit', 'bing',
 'taobao', 'csdn', 'live', 'weibo', 'sina', 'zoom', 'office', 'sohu', 'tmail', 'tiktok', 'canva', 'stackoverflow', 'naver', 'fandom', 'mail', 'myshopify',
 'douban' ]

vowels = ['a', 'e', 'i', 'o', 'u']

def domainProcessor(domain):
        ws.load()
        nb_suspect_keywords = 0
        nb_legitimate_keywords = 0
        v_ctr = 0
        c_ctr = 0
        txt = tldextract.extract(domain).suffix
        list_tld = txt.split(".")
        tld = list_tld[len(list_tld)-1]
        dom = domain[:-(len(tld)+1)]
        count = 0
        last_x = ""
        freq_transition_d_c = 0
        for k, g in groupby(domain) :
            nb = len(list(g))
            if (nb>1) :
                count += nb
        list_chars = re.findall('[²&éè_çà)=ù@^`#~&$АаБбВвГгДдЄєЖжЅѕꙂꙃЗзꙀꙁИиІіЇїКкЛлМмНнОоПпРрСсТтѸѹУуФфХхѠѡЦцЧчШшЩщЪъЫыЬьѢѣЮюꙖꙗѤѥѦѧЯяѪѫѨѩѬѭѮѯѰѱѲѳѴѵõɛ̃ûυ\-]*',domain)
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
            ratio_c_v = c_ctr/v_ctr
        except ZeroDivisionError:
            ratio_c_v = 70.0
        os.chdir("/home/cherifa/code_implementation/code_data/code_preprocessors")
        cpt_google = 0
        google_trends, twitter_trends = trends_extract()
        for trend in google_trends:
            for topic in trend:
                cpt_google = cpt_google+dom.count(topic)
        cpt = 0
        for trend in twitter_trends:
            for topic in trend:
                cpt = cpt+dom.count(topic)
        try:
            r = ns.nonsense(domain)
        except ValueError:
            r = True
        return [len(domain), tld, int(('.'+tld in suspect_keywords_list)) ,shannon_entropy(domain), nb_suspect_keywords, count, len(list_chars)- list_chars.count(''), len(ws.segment(dom)), sum(c.isdigit() for c in domain), int(r), len(set(domain)), nb_legitimate_keywords, domain.count('-'),domain.count('.') ,domain.find('-')/len(domain) ,freq_transition_d_c ,ratio_c_v, cpt_google, cpt]
def Testor(index, row):
                print(row['domain'], row['is_deleted'])
                domain = row['domain'][:-1]
                attr = domainProcessor(domain)
                attr_spam = attr
                attr_phish = attr
                attr_dga = attr
                attr_alexa = attr
                attr_gandi_selled = attr
                attr_gandi_non_value = attr
                new_le = LabelEncoder()
                tlds = new_le.fit_transform(np.append(df['tld'].to_numpy(),str(attr[1])))
                attr[1] = tlds[tlds.shape[0]-1]
                tlds_phishing = new_le.fit_transform(np.append(df_phishing['tld'].to_numpy(),str(attr[1])))
                attr_phish[1] = tlds_phishing[tlds_phishing.shape[0]-1]
                tlds_spams = new_le.fit_transform(np.append(df_spams['tld'].to_numpy(),str(attr[1])))
                attr_spam[1] = tlds_spams[tlds_spams.shape[0]-1]
                tlds_dgas = new_le.fit_transform(np.append(df_dgas['tld'].to_numpy(),str(attr[1])))
                attr_dga[1] = tlds[tlds.shape[0]-1]
                tlds_alexa = new_le.fit_transform(np.append(df_alexa['tld'].to_numpy(),str(attr[1])))
                attr_alexa[1] = tlds[tlds.shape[0]-1]
                tlds = new_le.fit_transform(np.append(df_gandi_non_value['tld'].to_numpy(),str(attr[1])))
                attr_gandi_non_value[1] = tlds[tlds.shape[0]-1]
                tlds = new_le.fit_transform(np.append(df_gandi_selled['tld'].to_numpy(),str(attr[1])))
                attr_gandi_selled[1] = tlds[tlds.shape[0]-1]
                attributes = da.reshape(da.from_array(attr),(1,19))
                attributes_phish = da.reshape(da.from_array(attr_phish),(1,19))
                attributes_spam = da.reshape(da.from_array(attr_spam),(1,19))
                attributes_dga = da.reshape(da.from_array(attr_dga),(1,19))
                attributes_alexa = da.reshape(da.from_array(attr_alexa),(1,19))
                attributes_gandi_selled = da.reshape(da.from_array(attr_gandi_selled),(1,19))
                attributes_gandi_non_value = da.reshape(da.from_array(attr_gandi_non_value),(1,19))
                prediction_elm = elm_from_joblib.predict(attributes)
                phishing_rate_onesvm = onesvm_from_pickle_phishing.predict(attributes_phish)
                spam_rate_onesvm = onesvm_from_pickle_spams.predict(attributes_spam)
                dga_rate_onesvm = onesvm_from_pickle_dgas.predict(attributes_dga)
                gandi_selled_rate_onesvm = onesvm_from_pickle_gandi_selled.predict(attributes_gandi_selled)
                alexa_rate_onesvm = onesvm_from_pickle_alexa.predict(attributes_alexa)
                gandi_non_value_rate_onesvm = onesvm_from_pickle_gandi_non_value.predict(attributes_gandi_non_value)
                malicious_rate_onesvm = phishing_rate_onesvm + spam_rate_onesvm + spam_rate_onesvm
                benign_rate_onesvm = alexa_rate_onesvm + gandi_non_value_rate_onesvm + gandi_selled_rate_onesvm
                if malicious_rate_onesvm > benign_rate_onesvm :
                        prediction_onesvm = 'True'
                elif malicious_rate_onesvm < benign_rate_onesvm :
                        prediction_onesvm = 'False'
                else :
                        prediction_onesvm = 'Suspect'
                with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
                        file1.write('ONESVM '+prediction_onesvm +'\n')
                with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
                        file1.write('ELM '+str(bool(np.round(prediction_elm)))+'\n')
                prediction_xgboost = xgb.dask.predict(client, loaded_model_xgboost,attributes)
                with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
                        file1.write('XGBOOST '+str(bool(prediction_xgboost.compute()[0]))+'\n')
                prediction_rf = loaded_model_rf.predict(attributes)
                with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
                        file1.write('RF '+str(bool(np.round(prediction_rf)))+'\n')
                phishing_rate = if_from_pickle_phishing.predict(attributes_phish)
                spam_rate = if_from_pickle_spams.predict(attributes_spam)
                dga_rate = if_from_pickle_dgas.predict(attributes_dga)
                gandi_selled_rate = if_from_pickle_gandi_selled.predict(attributes_gandi_selled)
                alexa_rate = if_from_pickle_alexa.predict(attributes_alexa)
                gandi_non_value_rate = if_from_pickle_gandi_non_value.predict(attributes_gandi_non_value)
                malicious_rate = phishing_rate + spam_rate + spam_rate
                benign_rate = alexa_rate + gandi_non_value_rate + gandi_selled_rate
                if malicious_rate>benign_rate :
                        prediction = 'True'
                elif malicious_rate<benign_rate :
                        prediction = 'False'
                else :
                        prediction = 'Suspect'
                with open("../../code_test/test_results/result_"+domain+".txt", "a") as file1:
                        file1.write('IF '+ prediction +'\n')
                os.chdir("../../code_test")
                return True
if __name__ == '__main__':
	print('Patience ... I am loading all your models and data :')
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	df_test = pd.read_csv('./query_data/query1.csv')
	df_test = df_test.reset_index()
	df = pd.read_csv('../code_data/data_all_lexical_features/all_domains_with_textual_attributes.csv')

	df_phishing = pd.read_csv('../code_data/instances_for_alikeness_test/instance_phish_alike.csv')

	df_spams = pd.read_csv('../code_data/instances_for_alikeness_test/instance_spam_alike.csv')
	df_dgas = pd.read_csv('../code_data/instances_for_alikeness_test/instance_dga_alike.csv')
	df_alexa = pd.read_csv('../code_data/instances_for_alikeness_test/instance_alexa_alike.csv')
	df_gandi_selled = pd.read_csv('../code_data/instances_for_alikeness_test/instance_gandi_selled_alike.csv')
	df_gandi_non_value = pd.read_csv('../code_data/instances_for_alikeness_test/instance_gandi_non_value_alike.csv')
	elm_from_joblib = joblib.load('../code_algorithms/algorithm_elmnn/models_overall/elm_spam_dga_phish_alexa_gandi_selled_gandi_non_value.pkl')
	onesvm_from_pickle_alexa = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_alexa_alike.pkl", "rb"))
	onesvm_from_pickle_gandi_selled = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_gandi_selled_alike.pkl", "rb"))
	onesvm_from_pickle_phishing = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_phish_alike.pkl", "rb"))
	onesvm_from_pickle_spams = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_spam_alike.pkl", "rb"))
	onesvm_from_pickle_dgas = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_dga_alike.pkl", "rb"))
	onesvm_from_pickle_gandi_non_value = pickle.load(open("../code_algorithms/algorithm_onesvm/models_alikeness/onesvm_gandi_non_value_alike.pkl", "rb"))
	if_from_pickle_alexa = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_alexa_alike.pkl", "rb"))
	if_from_pickle_gandi_selled = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_gandi_selled_alike.pkl", "rb"))
	if_from_pickle_phishing = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_phish_alike.pkl", "rb"))
	if_from_pickle_spams = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_spam_alike.pkl", "rb"))
	if_from_pickle_dgas = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_dga_alike.pkl", "rb"))
	if_from_pickle_gandi_non_value = pickle.load(open("../code_algorithms/algorithm_if/models_alikeness/if_gandi_non_value_alike.pkl", "rb"))
	loaded_model_rf = pickle.load(open("../code_algorithms/algorithm_rf/models_instances_for_overall/rf_spam_dga_phish_alexa_gandi_selled_gandi_non_value.pkl", "rb"))
	loaded_model_xgboost = pickle.load(open("../code_algorithms/algorithm_xgboost/models_instances_for_overall/xgboost_spam_dga_phish_alexa_gandi_selled_gandi_non_value.pkl", "rb"))
	print('Here we are ...')
	for index, row in df_test.iterrows():
		Testor(index, row)
