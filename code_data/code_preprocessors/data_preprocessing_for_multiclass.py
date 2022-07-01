import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import os
import shutil 

if __name__ == '__main__' :
	directories = ['instances_for_overall_test']
	for directory in directories :
		if os.path.exists('../preprocessed_multiclass_'+directory):
			shutil.rmtree('../preprocessed_multiclass_'+directory)
		os.mkdir('../preprocessed_multiclass_'+directory)
		for filename in os.listdir('../'+directory):
			f = os.path.join('../'+directory, filename)
			if os.path.isfile(f):
				start_time = time.time()
				df = pd.read_csv(f)
				df['nonesense'] = df['nonesense'].astype(int)
				df = df.drop(['domain_name'], axis=1)
				df = df.drop(['class'], axis=1)
				#df_processed = pd.get_dummies(df, prefix_sep="_",columns=['tld'])
				#label_encoders = {}
				new_le = LabelEncoder()
				df['tld'] = new_le.fit_transform(df['tld'])
				#label_encoders['tld'] = new_le
				#df_processed = pd.get_dummies(df, prefix_sep="_",columns=['type'])
				label_encoders = {}
				new_le = LabelEncoder()
				df['type'] = new_le.fit_transform(df['type'])
				label_encoders['type'] = new_le
				#df_processed = df_processed.drop(['type'], axis=1)
				df.to_csv('../preprocessed_multiclass_'+directory+'/preprocessed_'+filename)
				print("--- %s seconds ---" % (time.time() - start_time))
