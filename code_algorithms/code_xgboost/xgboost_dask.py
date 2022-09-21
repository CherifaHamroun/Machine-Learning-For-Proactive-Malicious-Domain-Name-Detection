import dask.dataframe as ddf
from dask_ml.model_selection import KFold,train_test_split
import time
import dask.distributed
import xgboost as xgb
import dask.array as da
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import pickle
import os
import shutil
from joblib import Parallel, delayed
from statistics import mean

def plot_roc_curve(directory,filename, fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("./performances_"+directory+"/ROC_"+filename+".png") #save as png

def KFoldValidation(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
        output = xgb.dask.train(
        client,
        {"eta":1,"gamma":0,"max_depth":10,"sampling_method":"gradient_based",
        "verbosity": 1, "tree_method": "hist", "objective": "binary:hinge"},
        dtrain,
        num_boost_round=10,
        evals=[(dtrain, "train")],)
        model = output['booster']
        y_pred = xgb.dask.predict(client, model, X_test).compute()
        global accuracy
        accuracy.append(skm.accuracy_score(y_test, y_pred))
        global f1_score
        f1_score.append(skm.f1_score(y_test, y_pred))
        global recall
        recall.append(skm.recall_score(y_test, y_pred))
        global precision
        precision.append(skm.precision_score(y_test, y_pred))
        #global roc_auc_score
        #roc_auc_score.append(skm.roc_auc_score(y_test, y_pred))
        global confusion
        confusion = confusion + skm.confusion_matrix(y_test, y_pred, labels = [0,1])
        return { "model":model, "accuracy":accuracy, "f1_score":f1_score,"recall":recall, "precision":precision,"confusion":confusion }
        #return {"model":model.best_estimator_, "score":model.best_score_}


if __name__ == "__main__":
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	directories = [ 'preprocessed_instances_for_overall_test']
	for directory in directories:
		if os.path.exists("./performances_"+directory[13:-5]):
			shutil.rmtree("./performances_"+directory[13:-5])
		os.mkdir('./performances_'+directory[13:-5])
		if os.path.exists("./models_"+directory[13:-5]):
			shutil.rmtree("./models_"+directory[13:-5])
		os.mkdir('./models_'+directory[13:-5])
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f) : #and filename == 'preprocessed_instance_phish_alexa.csv':
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				df = ddf.read_csv(f)
			#	print(df.shape)
				df = df.sample(frac=1)
				X = df.iloc[:, 3:22].values
				y = df.iloc[:, 22].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				print('--- Dataset splitting ...')
				start_time = time.time()
				#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
				kf = KFold(n_splits=2, random_state = 42, shuffle=True)
				print("--- Dataset splitted in  %s seconds ---" % (time.time() - start_time))
				accuracy = []
				f1_score = []
				precision = []
				recall = []
				roc_auc_score = []
				confusion = [[0,0],[0,0]]
				start_time = time.time()
				res = Parallel(n_jobs=os.cpu_count(), require='sharedmem')(delayed(KFoldValidation)(i,j) for i,j in kf.split(X))[0]
				#dtrain = xgb.dask.DaskDMatrix(client, x_train, y_train)
				#output = xgb.dask.train(
       				#client,
       				#{"eta":1,"gamma":0,"max_depth":100,"sampling_method":"gradient_based",
       				#"verbosity": 3, "tree_method": "hist", "objective": "binary:hinge"},
        			#dtrain,
        			#num_boost_round=100,
        			#evals=[(dtrain, "train")],)
				#booster = output['booster']
				#pickle.dump(booster, open('./models_'+directory[27:-5]+'/xgbdask_'+filename[22:-4]+'.pkl', "wb"))
				#prediction = xgb.dask.predict(client, booster, x_test)
				print('--- Training done in  ...', time.time() - start_time)
				#start_time = time.time()
				#y_pred = prediction.compute()
				#y_test = y_test.compute()
				pickle.dump(res["model"], open('./models_'+directory[13:-5]+'/xgboost_'+filename[22:-4]+'.pkl', "wb"))
				with open("./performances_"+directory[13:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					#file1.write('--- Accuracy : '+str(skm.accuracy_score(y_test, y_pred))+'\n')
					#file1.write('--- F1 score : '+str(skm.f1_score(y_test, y_pred, average='macro'))+'\n')
					#file1.write('--- Precision : '+str(skm.precision_score(y_test, y_pred, average='macro'))+'\n')
					#file1.write('--- Recall : '+str(skm.recall_score(y_test, y_pred, average='macro'))+'\n')
					#file1.write('--- ROC AUC score : '+str(skm.roc_auc_score(y_test, y_pred))+'\n')
					#fper, tper, thresholds = skm.roc_curve(y_test, y_pred) 
					#plot_roc_curve(directory[27:-5],filename[22:-4],fper,tper)
					#file1.write('--- Confusion Matrix : '+str(skm.confusion_matrix(y_test, y_pred, labels = [0,1]))+'\n')
					#file1.close()
					file1.write('--- Accuracy : '+str(mean(accuracy))+'\n')
					file1.write('--- F1 score : '+str(mean(f1_score))+'\n')
					file1.write('--- Precision : '+str(mean(precision))+'\n')
					file1.write('--- Recall : '+str(mean(recall))+'\n')
					#file1.write('--- ROC AUC score : '+str(mean(roc_auc_score))+'\n')
					file1.write('--- Confusion Matrix : '+str(confusion)+'\n')

