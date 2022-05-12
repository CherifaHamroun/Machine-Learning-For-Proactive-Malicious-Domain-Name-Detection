import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split
import time
import dask.array as da
from distributed import LocalCluster, Client
import xgboost as xgb
from dask_ml.model_selection import GridSearchCV
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

if __name__ == "__main__":
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            print('--- Loading your data ...')
            start_time = time.time()
            df = ddf.read_csv("preprocessed_data.csv")
            X = df.iloc[:, 2:11].values
            Y = df.iloc[:, 12].values
            print("--- Data loaded in  %s seconds ---" % (time.time() - start_time))
            print('--- Chunksizes computing ...')
            start_time = time.time()
            X.compute_chunk_sizes()
            Y.compute_chunk_sizes()
            print("--- Chunksizes computed in  %s seconds ---" % (time.time() - start_time))
            print('--- Dataset splitting ...')
            start_time = time.time()
            print("--- Dataset splitted in  %s seconds ---" % (time.time() - start_time))
            clf = xgb.dask.DaskXGBClassifier(n_estimators=100, tree_method="hist")
            clf.client = client
            model = GridSearchCV(clf, {'eta':[0,1],
				       #'max_depth': [2, 4],
                                       'n_estimators': [5, 10]},
					cv=2)
            model.fit(X, Y)
            print(model.best_params_)
            select = SelectFromModel(model, prefit=True)
            print(select)
            pickle.dump(model, open("clf.pickle.dat", "wb"))
            # some time later...

            # load model from file
            loaded_model = pickle.load(open("clf.pickle.dat", "rb"))
            # make predictions for test data
            predictions = loaded_model.predict(X)
            # evaluate predictions
            accuracy = accuracy_score(Y, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
