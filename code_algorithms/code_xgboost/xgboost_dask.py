import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split
import time
import dask.distributed
import xgboost as xgb
import dask.array as da
import numpy as np 
import sklearn.metrics as skm
import matplotlib.pyplot as plt 
import pickle
import os 

def plot_roc_curve(directory,filename, fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("./performances_"+directory+"/ROC_"+filename+".png") #save as png

if __name__ == "__main__":
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	directories = ['preprocessed_instances_for_overall_test','preprocessed_instances_for_alikeness_test']
	for directory in directories:
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f):
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				df = ddf.read_csv(f)
				#df = df.sample(frac=0.7)
				X = df.iloc[:, 3:18].values
				Y = df.iloc[:, 18].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				Y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				print('--- Dataset splitting ...')
				start_time = time.time()
				x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
				print("--- Dataset splitted in  %s seconds ---" % (time.time() - start_time))
				dtrain = xgb.dask.DaskDMatrix(client, x_train, y_train)
				output = xgb.dask.train(
        			client,
        			{"eta":1,"gamma":0,"max_depth":100,"sampling_method":"gradient_based",
				"verbosity": 3, "tree_method": "hist", "objective": "binary:hinge"},
        			dtrain,
        			num_boost_round=100,
        			evals=[(dtrain, "train")],)
				booster = output['booster']
				pickle.dump(booster, open('./models_'+directory[27:-5]+'/xgbdask_'+filename[22:-4]+'.pkl', "wb"))
				prediction = xgb.dask.predict(client, booster, x_test)
				print('--- Results computing ...')
				start_time = time.time()
				y_pred = prediction.compute()
				y_test = y_test.compute()
				with open("./performances_"+directory[27:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					file1.write('--- Accuracy : '+str(skm.accuracy_score(y_test, y_pred))+'\n')
					file1.write('--- F1 score : '+str(skm.f1_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- Precision : '+str(skm.precision_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- Recall : '+str(skm.recall_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- ROC AUC score : '+str(skm.roc_auc_score(y_test, y_pred))+'\n')
					fper, tper, thresholds = skm.roc_curve(y_test, y_pred) 
					plot_roc_curve(directory[27:-5],filename[22:-4],fper,tper)
					file1.write('--- Confusion Matrix : '+str(skm.confusion_matrix(y_test, y_pred, labels = [0,1]))+'\n')
					file1.close()
