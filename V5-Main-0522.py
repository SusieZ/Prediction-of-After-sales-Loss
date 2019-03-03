# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:16:14 2018

@author: Shugui
"""
import pandas as pd
import os,sys
sys.path.append("E:/Data_temp/After-Sales/Code/code/After_Sales_FC")
from Feature_set import Mainten_feature_list as mlist
from functions import Read_Raw as Rr
from functions import Feature_Processing as FP
from functions import Model_engineering as Me
from functions import Result_chart as RCrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#load raw       
inputfile=Rr.file_name(0)
df_mainten=Rr.read_raw(inputfiles=inputfile,sep1=',',head_num=0,Size=10000,code='gbk',iteration=True,looped=True,indexCol=False) 
#df_others
#df_wr

dfMat_view=Rr.df_view(df_mainten)    
# =============================================================================
# inputfile1='E:/Data_temp/After-Sales/Data/othrs_ct-0428.csv'
# df=pd.read_csv(inputfile1,nrows=1000,header=0,encoding='gbk',sep='\t')
# 
# =============================================================================

# feature processing: 
Processing= FP.feature_preprocessing(df_mainten,mlist.charCol,mlist.floatCol,"E:/Data_temp/After-Sales/Data/result_table/","Mat_Raw_0518.csv")   
Raw_data=Processing.feature_engineering()

# got model raw data
featurefile="E:/Data_temp/After-Sales/Data/result_table/Mat_Raw_0518.csv"
df_train= Rr.read_raw(inputfiles=featurefile,sep1=',',head_num=0,Size=10000,code='gbk',iteration=True,looped=True,indexCol=False)
    
# running model, got result   
dX_train,dX_test,dy_train,dy_test=Me.Train_test_data(df_train)
R_model,R_scores=Me.Model_all(dX_train,dX_test,dy_train,dy_test,'E:/Data_temp/After-Sales/model/k-models/my_model-0518-1.h5')
y_pred,t_scores=Me.Model_Useing('E:/Data_temp/After-Sales/model/k-models/my_model-0517-3.h5',dX_test,dy_test)
Me.Result_summary(y_pred,dy_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(list(dy_test),list(y_pred))
np.set_printoptions(precision=2)
class_names=[0,1]

# Plot non-normalized confusion matrix
plt.figure()
RCrt.plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
RCrt.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()