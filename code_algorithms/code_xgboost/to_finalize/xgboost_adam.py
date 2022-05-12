import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split
import time
import dask.distributed
import xgboost as xgb
import dask.array as da
import numpy as np 

if __name__ == "__main__":
	print('--- Loading your data ...')
	start_time = time.time()
	df = ddf.read_csv("preprocessed_data.csv")
	df = df.sample(frac=1)
	#df = dff.drop(['class'], axis=1)
	print(len(df))
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
	#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
	print("--- Dataset splitted in  %s seconds ---" % (time.time() - start_time))
	cluster = dask.distributed.LocalCluster()
	client = dask.distributed.Client(cluster)
	dtrain = xgb.DMatrix("preprocessed_data.csv")
	num_round = 2
	# do cross validation, this will print result out as
	# [iteration]  metric_name:mean_value+std_value
	# std_value is standard deviation of the metric
	param = {'max_depth':2, 'eta':1, 'objective':'binary:hinge'}
	xgb.cv(param, dtrain.compute(), num_round, nfold=5,
	metrics={'error'}, seed=0,
	callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)])

	# where X is a dask DataFrame or dask Array.
	#prediction = xgb.dask.predict(client, booster, x_test)
	np.savetxt('results.csv', prediction.compute(), delimiter=',')
	np.savetxt('true.csv', y_test.compute(), delimiter=',')
