# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:44:42 2018

@author: Shugui
"""
import os,sys
import pandas as pd

inputfilepath='E:/Data_temp/After-Sales/Data/'
file=['mainten_ct-0502.csv','othrs_ct-0502.csv','wr_ct-0516.csv']

def file_name(i):
    '''i within [0,1,2] ,the length equals to the file list'''
    inputfile1=os.path.join(inputfilepath,file[i])
    return inputfile1

def read_raw(inputfiles,sep1,head_num,Size,code,iteration,indexCol,looped=True):
    inputfile1=inputfiles
    loop=looped
    chunks=[]
    reader=pd.read_csv(inputfile1,sep=sep1,header=head_num,encoding=code,iterator=iteration,index_col=indexCol)
    while loop:
        try:
            chunk=reader.get_chunk(Size)
            chunks.append(chunk)
        except StopIteration:
            loop=False
            print('Iteration is stopped.')
    df=pd.concat(chunks,ignore_index=True)
    return df

def df_view(data):
    
    '''columns property/distribution/NA status 
    as well as NA rate within columns.'''
    
    df_type=pd.DataFrame(data.dtypes,columns=['type'])
    df_value=pd.DataFrame(data.describe()).T
    df_NA=pd.DataFrame(data.isnull().sum(),columns=['count_NA'])
    df_summ=pd.concat([df_type,df_value,df_NA],axis=1,join='outer')
    df_summ['obs']=data.shape[0]
    df_summ['na_rate']=df_summ['count_NA']/df_summ['obs']
    return df_summ