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

def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("ROC_curve_alexa_data.png") #save as png

if __name__ == "__main__":
	print('--- Loading your data ...')
	start_time = time.time()
	df = ddf.read_csv("../../code_data/preprocessed_data.csv")
	df = df.sample(frac=1)
	X = df.iloc[:, 2:14].values
	Y = df.iloc[:, 14].values
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
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	dtrain = xgb.dask.DaskDMatrix(client, x_train, y_train)
	output = xgb.dask.train(
        client,
        {"eta":1,"gamma":0,"max_depth":100,"sampling_method":"gradient_based",
	"verbosity": 3, "tree_method": "hist", "objective": "binary:hinge"},
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],)
	booster = output['booster']
	pickle.dump(booster, open("xgbdask_alexa_data.pickle.dat", "wb"))
	loaded_model = pickle.load(open("xgbdask_alexa_data.pickle.dat", "rb"))
	prediction = xgb.dask.predict(client, loaded_model, x_test)
	print('--- Results computing ...')
	start_time = time.time()
	predicted = prediction.compute()
	y = y_test.compute()
	print("--- Results computed in  %s seconds ---" % (time.time() - start_time))
	print('--- Accuracy : ',skm.accuracy_score(y, predicted))
	print('--- F1 score : ',skm.f1_score(y, predicted))
	print('--- Precision : ',skm.precision_score(y, predicted))
	print('--- Recall : ',skm.recall_score(y, predicted))
	print('--- ROC AUC score : ',skm.roc_auc_score(y, predicted))
	fper, tper, thresholds = skm.roc_curve(y, predicted) 
	plot_roc_curve(fper, tper)
	print('--- Confusion Matrix : ',skm.confusion_matrix(y, predicted, labels = [0,1]))

