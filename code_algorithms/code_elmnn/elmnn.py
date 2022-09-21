import numpy as np
import pandas as pd
import time 
import dask.dataframe as ddf
import dask.array as da
from dask_ml.model_selection import train_test_split
import  sklearn.metrics as skm
import joblib
import sklearn 
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import KFold, RandomizedSearchCV
from joblib import Parallel, delayed
from statistics import mean

def plot_roc_curve(directory,filename,fper, tper):  
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
        global model_previous
        model_previous = ELM(X_train.shape[1], 1, 92)
        model = model_previous
        model.train(X_train.compute(),y_train.compute().reshape(-1,1))
        model_previous = model
        y_pred = np.round(model.predict(X_test.compute()))
        global accuracy
        accuracy.append(skm.accuracy_score(y_test, y_pred))
        global f1_score
        f1_score.append(skm.f1_score(y_test, y_pred, average='micro'))
        global recall
        recall.append(skm.recall_score(y_test, y_pred, average='micro'))
        global precision
        precision.append(skm.precision_score(y_test, y_pred, average='micro'))
#       global roc_auc_score
#       roc_auc_score.append(skm.roc_auc_score(y_test, y_pred))
        global confusion
        confusion = confusion + skm.confusion_matrix(y_test, y_pred, labels = [0,1])
        return { "model":model_previous, "accuracy":accuracy, "f1_score":f1_score,"recall":recall, "precision":precision,"confusion":confusion }
        #return {"model":model.best_estimator_, "score":model.best_score_}

class ELM(object):  
    
    def __init__(self, inputSize, outputSize, hiddenSize):
        """
        Initialize weight and bias between input layer and hidden layer
        Parameters:
        inputSize: int
            The number of input layer dimensions or features in the training data
        outputSize: int
            The number of output layer dimensions
        hiddenSize: int
            The number of hidden layer dimensions        
        """    

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize       
        
        # Initialize random weight with range [-0.5, 0.5]
        self.weight = np.matrix(np.random.uniform(-0.5, 0.5, (self.hiddenSize, self.inputSize)))

        # Initialize random bias with range [0, 1]
        self.bias = np.matrix(np.random.uniform(-0.5, 0.5, (1, self.hiddenSize)))
        
        self.H = 0
        self.beta = 0

    def sigmoid(self, x):
        """
        Sigmoid activation function
        
        Parameters:
        x: array-like or matrix
            The value that the activation output will look for
        Returns:      
            The results of activation using sigmoid function
        """
        return 1 / (1 + np.exp(-1 * x))

    def predict(self, X):
        """
        Predict the results of the training process using test data
        Parameters:
        X: array-like or matrix
            Test data that will be used to determine output using ELM
        Returns:
            Predicted results or outputs from test data
        """
        X = np.matrix(X)
        y = self.sigmoid((X * self.weight.T) + self.bias) * self.beta

        return y.flatten().reshape(-1,1)

    def train(self, X, y):
        """
        Extreme Learning Machine training process
        Parameters:
        X: array-like or matrix
            Training data that contains the value of each feature
        y: array-like or matrix
            Training data that contains the value of the target (class)
        Returns:
            The results of the training process   
        """

        X = np.matrix(X)
        y = np.matrix(y)        
        
        # Calculate hidden layer output matrix (Hinit)
        self.H = (X * self.weight.T) + self.bias

        # Sigmoid activation function
        self.H = self.sigmoid(self.H)

        # Calculate the Moore-Penrose pseudoinverse matrix       
        H_moore_penrose = np.linalg.inv(self.H.T * self.H) * self.H.T

        # Calculate the output weight matrix beta
        self.beta = H_moore_penrose * y

        return self.H * self.beta
if __name__=='__main__':
	directories = ['preprocessed_instances_for_overall_test']
	for directory in directories:
		if os.path.exists("./performances_"+directory[27:-5]):
			shutil.rmtree("./performances_"+directory[27:-5])
		os.mkdir('./performances_'+directory[27:-5])
		if os.path.exists("./models_"+directory[27:-5]):
			shutil.rmtree("./models_"+directory[27:-5])
		os.mkdir('./models_'+directory[27:-5])
		for filename in os.listdir('../../code_data/'+directory):
			f = os.path.join('../../code_data/'+directory, filename)
			if os.path.isfile(f) :
				print('--- Loading '+filename+' data ...')
				start_time = time.time()
				dff = ddf.read_csv(f)
				dff = dff.sample(frac=1)
				df = dff.drop(['class'], axis=1)
				X = df.iloc[:, 3:22].values
				y = dff.iloc[:, 22].values
				print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
				print('--- Chunksizes computing ...')
				start_time = time.time()
				X.compute_chunk_sizes()
				y.compute_chunk_sizes()
				print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
				print('--- Dataset splitting ...')
				start_time = time.time()
				#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
				print("--- Dataset splitted in  %s seconds ---" % (time.time() - start_time))
				model_previous = ELM(X.shape[1], 1, 90)
				accuracy = []
				f1_score = []
				precision = []
				recall = []
				confusion = [[0,0],[0,0]]

				print('--- Training ...')
				start_time = time.time()
				#elm.train(x_train.compute(),y_train.compute().reshape(-1,1))
				kf = KFold(n_splits=2, random_state = 42, shuffle=True)
				res = Parallel(n_jobs=os.cpu_count(), require='sharedmem')(delayed(KFoldValidation)(i,j) for i,j in kf.split(X))[0]
				print("--- Training done in  %s seconds ---" % (time.time() - start_time))
				joblib.dump(res["model"], './models_'+directory[27:-5]+'/elm_'+filename[22:-4]+'.pkl')
				#y_pred = np.round(elm.predict(x_test.compute()))
				#y_test = y_test.compute()
				with open("./performances_"+directory[27:-5]+"/performance_"+filename[22:-4]+".txt", "w") as file1:
					file1.write('--- Accuracy : '+str(mean(accuracy))+'\n')
					file1.write('--- F1 score : '+str(mean(f1_score))+'\n')
					file1.write('--- Precision : '+str(mean(precision))+'\n')
					file1.write('--- Recall : '+str(mean(recall))+'\n')
#                                       file1.write('--- ROC AUC score : '+str(mean(roc_auc_score))+'\n')
					file1.write('--- Confusion Matrix : '+str(confusion)+'\n')

