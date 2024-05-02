import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import pickle


df1=pd.read_csv("finaldataset.csv", index_col=0)

x1 = df1.drop(columns='Customer Status')
y1 = df1["Customer Status"]
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.25, random_state=0)

# Random forest

Forest = RandomForestClassifier(max_depth=10,  min_samples_leaf=2, min_samples_split=8, n_estimators=400, random_state = 0)
Forest.fit(x_train1, y_train1)

# Gradien Boost
gc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=3, random_state = 0)
gc.fit(x_train1, y_train1)

#ensemble learning 
model = VotingClassifier(estimators=[ ('rf', Forest),('gb',model4)], voting='hard')
model.fit(x_train1,y_train1)
model.score(x_test1,y_test1)


# Make pickle file of our model
pickle.dump(model, open("model.pkl", "wb"))


