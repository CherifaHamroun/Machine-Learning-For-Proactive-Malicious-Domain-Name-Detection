import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time

if __name__ == '__main__' :
	start_time = time.time()
	df = pd.read_csv('data_textual_attributes.csv')
	df['nonesense'] = df['nonesense'].astype(int)
	df = df.drop(['domain_name'], axis=1)
	df_processed = pd.get_dummies(df, prefix_sep="__",columns=['tld'])
	label_encoders = {}
	new_le = LabelEncoder()
	df_processed['tld'] = new_le.fit_transform(df['tld'])
	label_encoders['tld'] = new_le
	df_processed.to_csv('preprocessed_data.csv')
	print("--- %s seconds ---" % (time.time() - start_time))
