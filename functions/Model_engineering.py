# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:48:29 2018

@author: Shugui
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.models import load_model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



def Train_test_data(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123,stratify=y)
    X_train=np.array(X_train).reshape(X_train.shape[0],1,X_train.shape[1]) 
    X_test=np.array(X_test).reshape(X_test.shape[0],1,X_test.shape[1]) 
    return X_train,X_test,y_train,y_test

def Model_all(X_train,X_test,y_train,y_test,epochs,batch_size,modelfile):
    model = Sequential()
    model.add(LSTM(128,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    hist=model.fit(X_train,y_train,epochs=200,batch_size=1000,validation_split=0.2,verbose=2,shuffle=True)
    print(hist.history)
    scores = model.evaluate(X_test, y_test, verbose = 1)
    print(scores)
    model.save(modelfile)
    return model,scores

def Model_Useing(modelfile,X_test,y_test):
    modell=load_model(modelfile)
    ld_scores=modell.evaluate(X_test,y_test,verbose=1)
    print(ld_scores)
    yp = modell.predict_classes(X_test).reshape(len(y_test))
    return yp,ld_scores


def Result_summary(yp,y_test):
    print('Accuracy Score:',accuracy_score(list(y_test),list(yp),normalize=True))
    print('Classification Report:\n',classification_report(list(y_test),list(yp)))
    print('Confusion Matrix:\n',confusion_matrix(list(y_test),list(yp),labels=[0,1]))
