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


# Define path using cookiecutter structure
data_path = os.path.join('models')
os.makedirs(data_path, exist_ok=True)  # prevent error if folder exists

# Save the model to models/model.pkl
model_path = os.path.join(data_path, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)