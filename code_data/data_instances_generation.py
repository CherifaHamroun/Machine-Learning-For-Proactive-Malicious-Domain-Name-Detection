import pandas as pd
import numpy as np
import getpass
from itertools import combinations

data_types = {
"spams" : {"length":51619, "position":1107181},
"dgas" : {"length":801667, "position":305514},
"phishing" : {"length":305514, "position":0},
"alexa" : {"length":1000000, "position":1658800},
"gandi_selled" : {"length":500000, "position":1158800},
}

if __name__ == '__main__' :
	#start_time = time.time()
	df = pd.read_csv('./all_domains_with_textual_attributes.csv')
	#types_malicious = input("Hello " + getpass.getuser()+ " So you wanna generate a dataset contaning what kind of malicious domains[ spams, phishing, dgas ] ? ( Please type your options as in the list and separate them with a blank) : ").split()
	#types_benign = input("What about benign domains [ alexa, gandi_selled, gandi_benign ] ? ( Please type your options as in the list and separate them with a blank) : ").split()
	#maliciousness = input("Do you want to detect a type of maliciousness ? [ yes, no ]")
	types_malicious = ['spams', 'dgas', 'phishing']
	types_benign = ['alexa','gandi_selled']
	list_combinations = list()
	for n in range(len(types_malicious) + 1):
		combi_malicious = list(combinations(types_malicious, n))
		for m in range(len(types_benign) + 1):
			combi_benign = list(combinations(types_benign, m))
			for mal in combi_malicious :
				for ben in combi_benign:
					minimum = 10000000
					dfObj = pd.DataFrame(columns=df.columns)
					my_malicious = list(mal)
					my_benign = list(ben)
					for type in my_malicious + my_benign :
						print(my_malicious +my_benign)
						if data_types[type]["length"] < minimum :
							minimum = data_types[type]["length"]
							essens = type
					if my_malicious and my_benign :
						for type in my_malicious:
							dfObj = pd.concat([dfObj,df[data_types[type]["position"]:data_types[type]["position"]+data_types[essens]["length"]]])
						for type in my_benign :
							dfObj = pd.concat([dfObj,df[data_types[type]["position"]:data_types[type]["position"]+int(data_types[essens]["length"]*len(my_malicious)/len(my_benign))]])	
						dfObj = dfObj.sample(frac=1).reset_index(drop=True)
						delim = "_"
						temp = list(map(str, my_malicious +my_benign ))
						res = delim.join(temp)
						print("dataset ",res," generated has shape ", dfObj.shape)
						dfObj.to_csv('./instances_for_overall_test/instance_'+str(res)+'.csv')
	for type in types_malicious+types_benign:
		dfObj = pd.DataFrame(columns=df.columns)
		new_df = pd.DataFrame(columns=df.columns)
		for other_type in types_malicious+types_benign:
			if (other_type is type):
				new_df = df[data_types[type]["position"]:data_types[type]["position"]+data_types[type]["length"]]
				new_df['class'] = [1]*new_df.shape[0]
				dfObj = pd.concat([dfObj,new_df])
			else :
				new_df = df[data_types[other_type]["position"]:data_types[other_type]["position"]+int(data_types[type]["length"]/(len(types_benign+types_malicious)-1))].copy()
				new_df['class'] = [0]*new_df.shape[0]
				dfObj = pd.concat([dfObj,new_df])
		dfObj = dfObj.sample(frac=1).reset_index(drop=True)
		print("dataset ",type," generated has shape ", dfObj.shape)
		dfObj.to_csv('./instances_for_alikeness_test/instance_'+type+'_alike.csv')
