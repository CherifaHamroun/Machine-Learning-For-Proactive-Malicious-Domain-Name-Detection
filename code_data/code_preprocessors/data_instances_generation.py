import pandas as pd
import numpy as np
import getpass
from itertools import combinations
import os 
import shutil

data_types = {
"spam" : {"length":51619, "position":1107181},
"dga" : {"length":801667, "position":305514},
"phish" : {"length":305514, "position":0},
"alexa" : {"length":1000000, "position":1658800},
"gandi_selled" : {"length":500000, "position":1158800},
"gandi_non_value":{"length":500000, "position":1158800},
}

if __name__ == '__main__' :
	if os.path.exists("../instances_for_overall_test"):
		shutil.rmtree("../instances_for_overall_test")
	os.mkdir('../instances_for_overall_test')
	if os.path.exists("../instances_for_alikeness_test"):
		shutil.rmtree("../instances_for_alikeness_test")
	os.mkdir('../instances_for_alikeness_test')
	#start_time = time.time()
	#with open('../data_all_lexical_features/valid_domains_lists_length.txt', "r") as fd:
        #	list_lens = fd.read().splitlines()
	df = pd.read_csv('../data_all_lexical_features/all_domains_with_textual_attributes.csv')
	data_types['spam'] = {"length":df['type'].value_counts()['spam'], "position": df.index[df['type'] == 'spam'].tolist()[0]}
	data_types['phish'] = {"length":df['type'].value_counts()['phish'], "position":df.index[df['type'] == 'phish'].tolist()[0]}
	data_types['dga'] = {"length":df['type'].value_counts()['dga'], "position":df.index[df['type'] == 'dga'].tolist()[0]}
	data_types['alexa'] = {"length":df['type'].value_counts()['alexa'], "position":df.index[df['type'] == 'alexa'].tolist()[0]}
	data_types['gandi_selled'] = {"length":df['type'].value_counts()['gandi_selled'], "position":df.index[df['type'] == 'gandi_selled'].tolist()[0]}
	data_types['gandi_non_value'] = {"length":df['type'].value_counts()['gandi_non_value'], "position":df.index[df['type'] == 'gandi_non_value'].tolist()[0]}
	#df = pd.read_csv('../data_all_lexical_features/all_domains_with_textual_attributes.csv')
	#types_malicious = input("Hello " + getpass.getuser()+ " So you wanna generate a dataset contaning what kind of malicious domains[ spams, phishing, dgas ] ? ( Please type your options as in the list and separate them with a blank) : ").split()
	#types_benign = input("What about benign domains [ alexa, gandi_selled, gandi_benign ] ? ( Please type your options as in the list and separate them with a blank) : ").split()
	#maliciousness = input("Do you want to detect a type of maliciousness ? [ yes, no ]")
	types_malicious = ['spam', 'dga', 'phish']
	types_benign = ['alexa','gandi_selled', 'gandi_non_value']
	list_combinations = list()
	for n in range(len(types_malicious) + 1):
		combi_malicious = list(combinations(types_malicious, n))
		for m in range(len(types_benign) + 1):
			combi_benign = list(combinations(types_benign, m))
			for mal in combi_malicious :
				for ben in combi_benign:
					minimum_malicious = 10000000
					minimum_benign = 10000000
					dfObj = pd.DataFrame(columns=df.columns)
					my_malicious = list(mal)
					my_benign = list(ben)
					for type_malicious in my_malicious :
						if data_types[type_malicious]["length"] < minimum_malicious :
							minimum_malicious = data_types[type_malicious]["length"]
							essens_malicious = type_malicious
					for type_benign in my_benign :
						if data_types[type_benign]["length"] < minimum_benign :
							minimum_benign = data_types[type_benign]["length"]
							essens_benign = type_benign
					if my_malicious and my_benign :
						minimum = min(data_types[essens_malicious]["length"]*len(my_malicious), data_types[essens_benign]["length"]*len(my_benign))
						for type in my_malicious:
							dfObj = pd.concat([dfObj,df[data_types[type]["position"]:data_types[type]["position"]+int(minimum/len(my_malicious))]])
						for type in my_benign :
							dfObj = pd.concat([dfObj,df[data_types[type]["position"]:data_types[type]["position"]+int(minimum/len(my_benign))]])
						dfObj = dfObj.sample(frac=1).reset_index(drop=True)
						delim = "_"
						temp = list(map(str, my_malicious +my_benign ))
						res = delim.join(temp)
						print("dataset ",res," generated has shape ", dfObj.shape)
						dfObj.to_csv('../instances_for_overall_test/instance_'+str(res)+'.csv')
	for type in types_malicious+types_benign:
		dfObj = pd.DataFrame(columns=df.columns)
		new_df = pd.DataFrame(columns=df.columns)
		#for other_type in types_malicious+types_benign:
		#	if (other_type is type):
		#new_df = df[data_types[type]["position"]:data_types[type]["position"]+data_types[type]["length"]]
		new_df = df.loc[df.type == type]
		new_df['class'] = [1]*new_df.shape[0]
		dfObj = pd.concat([dfObj,new_df])
			#else :
			#	new_df = df[data_types[other_type]["position"]:data_types[other_type]["position"]+int(data_types[type]["length"] /(5*len(types_benign+types_malicious)-1))].copy()
			#	new_df['class'] = [0]*new_df.shape[0]
			#	dfObj = pd.concat([dfObj,new_df])
		dfObj = dfObj.sample(frac=1).reset_index(drop=True)
		print("dataset ",type," generated has shape ", dfObj.shape)
		dfObj.to_csv('../instances_for_alikeness_test/instance_'+type+'_alike.csv')
