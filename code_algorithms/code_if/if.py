import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split, RandomizedSearchCV, KFold
import time
import dask.array as da
import numpy as np 
import sklearn.metrics as skm
import matplotlib.pyplot as plt 
import pickle
from sklearn.ensemble import IsolationForest
import os 
import shutil
from joblib import Parallel, delayed
from scipy.stats import uniform, truncnorm, randint
from statistics import mean
import gc

def KFoldValidation(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        global model_previous
        model = model_previous
        model = RandomizedSearchCV(model, model_params, n_iter=2, cv=2, random_state=42, scoring ="accuracy" )
       	model.fit(X_train, y_train)
        #model = model.best_estimator_
        model_previous = model
        #return {"model":model.best_estimator_, "score":model.best_score_}
        y_pred = model.predict(X_test)
        global accuracy
        accuracy.append(model.best_score_)
        #global f1_score
        #f1_score.append(skm.f1_score(y_test, y_pred))
        #global recall
        #recall.append(skm.recall_score(y_test, y_pred, average = 'micro'))
        #global precision
        #precision.append(skm.precision_score(y_test, y_pred, average = 'micro'))
        #global roc_auc_score
        #roc_auc_score.append(skm.roc_auc_score(y_test, y_pred, average = 'micro'))
        #global confusion
        #confusion = confusion + skm.confusion_matrix(y_test, y_pred, labels = [0,1])
        #return { "model":model, "accuracy":accuracy, "f1_score":f1_score,"recall":recall, "precision":precision,"roc_auc_score":roc_auc_score,"confusion":confusion }
        return { "model":model_previous, "accuracy":accuracy}
if __name__ == "__main__":
	directories = ['preprocessed_instances_for_alikeness_test']
	for directory in directories:
		if os.path.exists("./performances_"+directory[27:-5]):
			shutil.rmtree("./performances_"+directory[27:-5])
		os.mkdir("./performances_"+directory[27:-5])
		if os.path.exists('./models_'+directory[27:-5]):
			shutil.rmtree('./models_'+directory[27:-5])
		os.mkdir('./models_'+directory[27:-5])
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f):
				gc.collect()
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				df = ddf.read_csv(f)
				df = df.sample(frac=1)
				X = df.iloc[:, 3:22].values
				y = df.iloc[:, 22].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				kf = KFold(n_splits=2, random_state = 42, shuffle=True)
				print('--- Training ---')
				model_params = {
				#'n_estimators':randint(100,150),
                                'random_state': randint(0,50),
				'max_samples' : uniform(0.01, 0.199),
                                #'max_features': truncnorm(a=0.5, b=1, loc=0.25, scale=0.1),
                                }
				accuracy = []
				start_time = time.time()
				model_previous = IsolationForest(n_estimators = 100, warm_start=True, n_jobs = -1, verbose = 1)
				res = Parallel(n_jobs=os.cpu_count(), require='sharedmem')(delayed(KFoldValidation)(i,j) for i,j in kf.split(X))[0]
				print('--- Training done in %s seconds' % (time.time() - start_time))
				pickle.dump(res["model"], open('./models_'+directory[27:-5]+'/if_'+filename[22:-4]+'.pkl', "wb"))
				with open("./performances_"+directory[27:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					file1.write('--- Accuracy : '+str(mean(res["accuracy"]))+'\n')
