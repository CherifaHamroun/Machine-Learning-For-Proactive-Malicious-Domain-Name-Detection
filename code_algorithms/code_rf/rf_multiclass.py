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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
import os 
from statistics import mean
from joblib import Parallel, delayed
from pprint import pprint
from scipy.stats import uniform, truncnorm, randint
import shutil

def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("ROC_curve_alexa_data.png") #save as png

def KFoldValidation(train_index, test_index):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	model = RandomForestClassifier()
	model = RandomizedSearchCV(model, model_params, n_iter=10, cv=2, random_state=42)
	model.fit(X_train, y_train)
	model = model.best_estimator_
	y_pred = model.predict(X_test)
	global accuracy
	accuracy.append(skm.accuracy_score(y_test, y_pred))
	global f1_score
	f1_score.append(skm.f1_score(y_test, y_pred, average='micro'))
	global recall
	recall.append(skm.recall_score(y_test, y_pred, average='micro'))
	global precision
	precision.append(skm.precision_score(y_test, y_pred, average='micro'))
	#global roc_auc_score
	#roc_auc_score.append(skm.roc_auc_score(y_test, y_pred, average='micro'))
	global confusion
	confusion = confusion + skm.multilabel_confusion_matrix(y_test, y_pred, labels = [0,1,2,3,4])
	return { "model":model, "accuracy":accuracy, "f1_score":f1_score,"recall":recall, "precision":precision,"roc_auc_score":roc_auc_score,"confusion":confusion }
	#return {"model":model.best_estimator_, "score":model.best_score_}

if __name__ == "__main__": 
	directories = ['preprocessed_multiclass_instances_for_overall_test']
	for directory in directories:
		if os.path.exists('./models_'+directory[13:-5]):
			shutil.rmtree('./models_'+directory[13:-5])
		os.mkdir('./models_'+directory[13:-5])
		if os.path.exists('./performances_'+directory[13:-5]):
			shutil.rmtree('./performances_'+directory[13:-5])
		os.mkdir('./performances_'+directory[13:-5])
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f) and filename=="preprocessed_instance_spams_dgas_phishing_alexa_gandi_selled_gandi_non_value.csv":
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				df = ddf.read_csv(f)
				df = df.sample(frac=1)
				X = df.iloc[:, 3:20].values
				y = df.iloc[:, 20].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				kf = KFold(n_splits=2, random_state = 42, shuffle=True)
				print('--- Training ---')
				start_time = time.time()
				model_params = {
				# randomly sample numbers from 4 to 204 estimators
				'n_estimators': randint(4,200),
				# normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
				'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
				# uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
				'min_samples_split': uniform(0.01, 0.199)
				}
				accuracy = []
				f1_score = []
				precision = []
				recall = []
				roc_auc_score = []
				confusion = [[0,0],[0,0]]
				res = Parallel(n_jobs=os.cpu_count(), require='sharedmem')(delayed(KFoldValidation)(i,j) for i,j in kf.split(X))[0]
				#rf_model = RandomForestClassifier()
				#model = RandomizedSearchCV(rf_model, model_params,n_jobs = os.cpu_count(),  n_iter=100, cv=5, random_state=1)
				#model.fit(X, y)
				print('--- Training done in %s seconds' % (time.time() - start_time))
				pickle.dump(res["model"], open('./models_'+directory[13:-5]+'/rf_'+filename[22:-4]+'.pkl', "wb"))
				#pprint(model.best_estimator_.get_params())
				with open("./performances_"+directory[13:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					#file1.write('--- Accuracy : '+str(res["score"])+'\n')
					file1.write('--- Accuracy : '+str(mean(accuracy))+'\n')
					file1.write('--- F1 score : '+str(mean(f1_score))+'\n')
					file1.write('--- Precision : '+str(mean(precision))+'\n')
					file1.write('--- Recall : '+str(mean(recall))+'\n')
					#file1.write('--- ROC AUC score : '+str(mean(roc_auc_score))+'\n')
					file1.write('--- Confusion Matrix : '+str(confusion)+'\n')
