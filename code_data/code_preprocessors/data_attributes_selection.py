import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from numpy import array

df = pd.read_csv('../preprocessed_instances_for_overall_test/preprocessed_instance_spam_dga_phish_alexa_gandi_selled_gandi_non_value.csv')
df = df.sample(frac=1)
#df = df.drop(['last_hyphend_position'], axis=1)
X = df.iloc[:, 3:15].values
y = df.iloc[:, 15].values
select = SelectKBest(score_func=chi2, k=8)
z = select.fit_transform(X,y)
filter = select.get_support()
features = array(df.columns)
print("All features:")
print(features)
print("Selected best 3:")
print(filter)
print(z) 
