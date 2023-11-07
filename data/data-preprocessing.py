%matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from torch.nn import init
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

path = ".\data.xlsx"
data_df = pd.read_excel(path)
data_df.to_csv('data.csv', index=None, header=True)
df = pd.DataFrame(pd.read_csv("data.csv"))

x_data = data_df[[i for i in data_df.columns if i not in ['KT2','KL2']]]
y1_data = data_df.KT2
y2_data = data_df.KL2

feature_names = [i for i in data_df.columns if i not in ['KT2','KL2']]

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

x1_train, x1_test, y1_train, y1_test = train_test_split(x_data, y1_data, test_size=0.2, random_state=1)

for i in range(0,15):
    amean = x1_train[:,i].mean()
    # print(amean)
    astd = x1_train[:,i].std()
    # print(astd)
    x1_train[:,i] = (x1_train[:,i]-amean)/astd
    x1_test[:,i] = (x1_test[:,i]-amean)/astd
    x_data[:,i]= (x_data[:,i]-amean)/astd

x_data=pd.DataFrame(x_data)
x_data.columns = feature_names
y1_data=pd.DataFrame(y1_data)
y1_data.columns = ['KT2']
y2_data=pd.DataFrame(y2_data)
y2_data.columns = ['KL2']
Data=pd.concat([x_data,y1_data,y2_data],axis=1)

Data[['KT2','KL2']] = Data[['KT2','KL2']] / 1000

Data.to_csv('data_processed.csv', index=None, header=True)