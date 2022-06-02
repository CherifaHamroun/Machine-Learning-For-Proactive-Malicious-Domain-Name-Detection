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
from sklearn.model_selection import KFold
import os 

def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("ROC_curve_alexa_data.png") #save as png

if __name__ == "__main__":
	directories = ['preprocessed_instances_for_overall_test', 'preprocessed_instances_for_alikeness_test']
	for directory in directories:
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f):
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				df = ddf.read_csv(f)
				df = df.sample(frac=0.8)
				X = df.iloc[:, 3:18].values
				y = df.iloc[:, 18].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				kf = KFold(n_splits=2, random_state = 42, shuffle=True)
				accuracy = []
				precision = []
				recall = []
				roc_auc_score = []
				f1_score = []
				confusion_matrix = [[0,0], [0,0]]
				print('--- Training ---')
				start_time = time.time()
				for train_index, test_index in kf.split(X):
					X_train, X_test = X[train_index], X[test_index]
					y_train, y_test = y[train_index], y[test_index]
					model = RandomForestClassifier(n_estimators = 100, random_state = 24)
					model.fit(X_train, y_train)
					y_pred = model.predict(X_test)
					accuracy.append(skm.accuracy_score(y_test, y_pred))
					f1_score.append(skm.f1_score(y_test, y_pred))
					recall.append(skm.recall_score(y_test, y_pred))
					precision.append(skm.precision_score(y_test, y_pred))
					roc_auc_score.append(skm.roc_auc_score(y_test, y_pred))
					confusion_matrix = confusion_matrix + skm.confusion_matrix(y_test, y_pred, labels = [0,1])
				print('--- Training done in %s seconds' % (time.time() - start_time))
				pickle.dump(model, open('./models_'+directory[27:-5]+'/rf_'+filename[22:-4]+'.pkl', "wb"))
				with open("./performances_"+directory[27:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					file1.write('--- Accuracy : '+str(skm.accuracy_score(y_test, y_pred))+'\n')
					file1.write('--- F1 score : '+str(skm.f1_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- Precision : '+str(skm.precision_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- Recall : '+str(skm.recall_score(y_test, y_pred, average='macro'))+'\n')
					file1.write('--- ROC AUC score : '+str(skm.roc_auc_score(y_test, y_pred))+'\n')
					fper, tper, thresholds = skm.roc_curve(y_test, y_pred) 
					file1.write('--- Confusion Matrix : '+str(confusion_matrix)+'\n')
