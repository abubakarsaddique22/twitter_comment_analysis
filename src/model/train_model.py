import numpy as np 
import pandas as pd 
import os 
import pickle 
from sklearn.ensemble import GradientBoostingClassifier
import yaml

params=yaml.safe_load(open('params.yaml','r'))['train_model']


train_data=pd.read_csv('data/interim/train_transform.csv')

X_train=train_data.iloc[:,:-1].values
y_train=train_data.iloc[:,-1].values 


clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])
clf.fit(X_train, y_train)


model_path = os.path.join('models', 'model.pkl')
os.makedirs('models', exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
