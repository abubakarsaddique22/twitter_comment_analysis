import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score,classification_report
import json
import os 


# load data 
with open('models/model.pkl','rb') as f:
    clf=pickle.load(f)
test_data=pd.read_csv('data/interim/test_transform.csv')

X_test=test_data.iloc[:,:-1].values
y_test=test_data.iloc[:,-1].values 

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_pred)


metric_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}



metrics_path = os.path.join('reports', 'metrics.json')
os.makedirs(os.path.dirname(metrics_path), exist_ok=True) 
with open(metrics_path, 'w') as f:
    json.dump(metric_dict,f,indent=4)