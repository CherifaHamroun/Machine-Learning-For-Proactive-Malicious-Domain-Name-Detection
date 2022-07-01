import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
from sklearn import preprocessing
import os
import shutil

if __name__ == '__main__' :
	directories = ['instances_for_overall_test','instances_for_alikeness_test']
	for directory in directories :
		if os.path.exists('../preprocessed_'+directory):
			shutil.rmtree('../preprocessed_'+directory)
		os.mkdir('../preprocessed_'+directory)
		for filename in os.listdir('../'+directory):
			f = os.path.join('../'+directory, filename)
			if os.path.isfile(f):
				start_time = time.time()
				df = pd.read_csv(f)
				df['nonesense'] = df['nonesense'].astype(int)
				df_processed = df.drop(['domain_name'], axis=1)
				#df_processed = pd.get_dummies(df, prefix_sep="_",columns=['type'])
				#label_encoders = {}
				new_le = LabelEncoder()
				df_processed['tld'] = new_le.fit_transform(df['tld'])
				#df_processed['class'] = new_le.fit_transform(df['type'])
				#label_encoders['type'] = new_le
				df_class = df_processed.drop(['type'], axis=1)
				df_class.to_csv('../preprocessed_'+directory+'/preprocessed_'+filename)
				print("--- Dataset prerocessed in  %s seconds ---" % (time.time() - start_time))
