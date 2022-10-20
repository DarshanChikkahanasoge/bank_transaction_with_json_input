import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


data=pd.read_csv('bank_transaction/transaction1.csv')

to_update=data.copy()
del to_update['TransactionDate']

x=to_update.drop(['IsFraud'],axis=1)
y=to_update['IsFraud']

num_pipeline = Pipeline([
    ('std', StandardScaler())
])


cat_attribs=["TransactionID"]
num_attribs=list(x)
num_attribs.remove('TransactionID')
#print(num_attribs)

full_pipeline = ColumnTransformer([
    ("cat",OrdinalEncoder(),cat_attribs),
    ("num",num_pipeline,num_attribs),
])
#print(x)
prepared_data=full_pipeline.fit_transform(x)
#print(prepared_data)


xtrain,xtest,ytrain,ytest=train_test_split(np.array(prepared_data),np.array(y),test_size=0.3,shuffle=True)




from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,
                                verbose=1,class_weight="balanced")

rf_clf.fit(xtrain,ytrain)


with open("bank_transaction/model.bin",'wb') as f_out:
    pickle.dump(rf_clf,f_out)
    f_out.close()