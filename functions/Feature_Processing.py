# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:47:06 2018

@author: Shugui
"""
import pandas as pd
import numpy as np
import os,sys
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class feature_preprocessing:
    '''
    data cleaning / processing
    '''
    def __init__(self,data,charColumns,floatColumns,savepath,filename):
        
        #self.le=sklearn.preprocessing.LabelEncoder()
        #self.Scaler=sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
        self.df=data
        self.data=self.df.dropna(subset=['IS_LEAVE'])
        self.charCol=charColumns
        self.floatCol=floatColumns
        self.fea_col=self.charCol+self.floatCol
        self.savepath=savepath
        self.filename=filename
        #self.le=sklearn.preprocessing.LabelEncoder()
        #self.Scaler=sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
        self.dfCTrs_Scal=self.facCol_preprocessing()
        self.dffl_Scal= self.numCol_preprocessing()
        self.dataset=self.feature_table()
        #self.savepath=savepath
        #self.filename=filename
        
    def facCol_preprocessing(self):
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df_char=self.data[self.charCol].astype(str)
        dfCTrs=df_char.apply(le.fit_transform)
        sc_fac=MinMaxScaler(feature_range=(0,1))
        dfCTrs_Scal=pd.DataFrame(sc_fac.fit_transform(dfCTrs),columns=self.charCol)
        #dfCTrs_Scal=pd.DataFrame(self.Scaler.fit_transform(dfCTrs),columns=self.charCol)
        return dfCTrs_Scal
    
    def numCol_preprocessing(self):
        scaler=MinMaxScaler(feature_range=(0,1))
        df_f0=self.data[self.floatCol].fillna(0)
        dffl_Scal=pd.DataFrame(scaler.fit_transform(df_f0),columns=self.floatCol)
        #dffl_Scal=pd.DataFrame(self.Scaler.fit_transform(df_f0),columns=self.floatCol)
        return dffl_Scal

    def feature_table(self):
        y_temp=np.where(self.data.loc[:,'IS_LEAVE']=='Y',1,0).reshape(-1,1)
        df_y=pd.DataFrame(y_temp,columns=['YY'])
        df_y=df_y.reset_index(drop=True)
        datasetb=pd.concat([self.dfCTrs_Scal,self.dffl_Scal,df_y],axis=1)
        return datasetb
    
    def feature_engineering(self):
        vin_freq=pd.DataFrame(self.dataset.VIN.value_counts())
        datax=[]
        datay=[]
        vin_list=vin_freq.loc[vin_freq['VIN']>=6,:].index.tolist()
        for vins in vin_list:
            temptb=self.dataset.loc[self.dataset['VIN']==vins,:]
            for i in range(temptb.shape[0]-6):
                dx=list(np.array(temptb[self.fea_col].iloc[i:i+6,:]).reshape(1,-1))
                dy=list(temptb['YY'].iloc[i+5:i+6])
                datax.append(dx)
                datay.append(dy)
        datax=np.array(datax).reshape(len(datax),-1) 
        X_input=pd.DataFrame(datax)       
        Y=pd.DataFrame(datay)  
        self.input_Raw=pd.concat([X_input,Y],axis=1)
        savefile=os.path.join(self.savepath,self.filename)
        self.input_Raw.to_csv(savefile,index=False)
        return self.input_Raw