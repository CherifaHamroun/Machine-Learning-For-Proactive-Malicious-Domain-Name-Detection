import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
from sklearn import preprocessing
import os
if __name__ == '__main__' :
	directories = ['instances_for_overall_test','instances_for_alikeness_test']
	for directory in directories :
		for filename in os.listdir(directory):
			f = os.path.join(directory, filename)
			if os.path.isfile(f):
				start_time = time.time()
				df = pd.read_csv(f)
				df['nonesense'] = df['nonesense'].astype(int)
				df = df.drop(['domain_name'], axis=1)
			#df_processed = pd.get_dummies(df, prefix_sep="__",columns=['tld'])
			#label_encoders = {}
				df_processed = df
				new_le = LabelEncoder()
				df_processed['tld'] = new_le.fit_transform(df['tld'])
			#label_encoders['tld'] = new_le
			#min_max_scaler = preprocessing.MinMaxScaler()
			#X_scaled = min_max_scaler.fit_transform(df_processed.values)
			#df_processed = pd.DataFrame(X_scaled)
			#df_processed.drop('entropy', inplace=True, axis=1)
				df_processed.to_csv('./preprocessed_'+directory+'/preprocessed_'+filename)
				print("--- Dataset prerocessed in  %s seconds ---" % (time.time() - start_time))
