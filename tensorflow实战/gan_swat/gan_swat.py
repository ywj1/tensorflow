import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('D:/SWaT_Dataset_100000.csv').drop('Timestamp',axis=1).drop('Class',axis=1)


tf.set_random_seed(1)
#设置参数
dim_input=51
D_dim=[dim_input,100,50,2]
G_dim=[50,100,dim_input]
def get_generator(noise,n_units,out_dim,reuse=False):
    '''
    生成器
    noise:噪声
    n_units:隐元个数
    out_dim:输出层数
    '''
    
