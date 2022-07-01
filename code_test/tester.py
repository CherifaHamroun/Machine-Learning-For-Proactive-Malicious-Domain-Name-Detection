import pandas as pd
import os
import shutil

df = pd.read_csv('query.csv')
df = df.fillna(0)[['domain', 'is_deleted']]
print(df)
df = df.reset_index()
if os.path.exists("./test_results"):
	shutil.rmtree("./test_results")
os.mkdir('./test_results')
for index, row in df.iterrows():
	print(row['domain'], row['is_deleted'])
	domain = row['domain'][:-1]
	os.chdir("../code_algorithms/algorithm_rf/")
	os.system("python3 predictor.py "+domain)
	os.chdir("../algorithm_elmnn")
	os.system("python3 predictor.py "+domain)
	os.chdir("../algorithm_xgboost")
	os.system("python3 predictor.py "+domain)
	os.chdir("../algorithm_onesvm")
	os.system("python3 predictor.py "+domain)
	os.chdir("../algorithm_if")
	os.system("python3 predictor.py "+domain)
	os.chdir("../../code_test")
	os.system("cat ./test_results/result_"+domain+".txt")
	#with open('./test_results/result_'+domain+'.txt', "r") as fd:
	#	list_predictions = fd.read().splitlines()
	#with open('supp_mal.txt', 'w') as f:
	#	print(domain,  file=f)
	#with open('supp_mal.txt', 'w') as f:
        #	print(domain,  file=f)
	#os.system("rm result.txt")
