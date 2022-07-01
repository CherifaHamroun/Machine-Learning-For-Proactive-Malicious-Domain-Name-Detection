import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
from sklearn import preprocessing

if __name__ == '__main__' :
	start_time = time.time()
	df = pd.read_csv('data_textual_attributes.csv')
	df['nonesense'] = df['nonesense'].astype(int)
	df = df.drop(['domain_name'], axis=1)
	#df_processed = pd.get_dummies(df, prefix_sep="__",columns=['tld'])
	#label_encoders = {}
	df_processed = df

	new_le = LabelEncoder()
	df_processed['tld'] = new_le.fit_transform(df['tld'])
	#label_encoders['tld'] = new_le
	min_max_scaler = preprocessing.MinMaxScaler()
	X_scaled = min_max_scaler.fit_transform(df_processed.values)
	#df_processed = pd.DataFrame(X_scaled)
	#df_processed.drop('entropy', inplace=True, axis=1)
	#df_processed.drop('hyphend_position', inplace=True, axis=1)
	#df_processed.drop('nonesense', inplace=True, axis=1)
	df_processed.to_csv('preprocessed_data.csv')
	print("--- %s seconds ---" % (time.time() - start_time))
