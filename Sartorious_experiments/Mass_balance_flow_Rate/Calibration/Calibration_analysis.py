#%%
import pandas as pd
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
%matplotlib qt
import os
import regex as re
from sklearn.linear_model import LinearRegression
#%% Batch process 
files_dict = {}
folder_name = r'2023-03-10'
for file in os.listdir(folder_name):
    if '.csv' in file:
        speed = re.split(r'[.\s]\s*',file)[-2]+ '_'+re.split(r'[.\s]\s*',file)[1]
        files_dict[speed] = pd.read_csv(folder_name+r'/'+file,index_col=0)

for key in files_dict:
    files_dict[key]['ts'] = files_dict[key]['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
    files_dict[key]['ts']=files_dict[key]['ts']-files_dict[key]['ts'][0]
    # files_dict[key]['Mass']=files_dict[key]['Mass']-files_dict[key]['Mass'][0]
    files_dict[key]['Mass_analysis_smooth']= signal.savgol_filter(files_dict[key]['Mass'],91,1)
    files_dict[key]['Mass_analysis_derivative_smooth']=files_dict[key]['Mass_analysis_smooth'].diff()
    
#%%Batch Analisisof aspiration curves by mass

for key in files_dict:
    measured = files_dict[key].where(files_dict[key]['ts']>1).dropna().where(files_dict[key]['Mass']>-990)
    data = files_dict[key].where(files_dict[key]['ts']>1).dropna().where(files_dict[key]['Mass'].between(-800,-200)).dropna()
    X = data.loc[:, 'ts'].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.loc[:, 'Mass_change'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)
    slope_dict_1[key]=linear_regressor.coef_
    y_hat  = linear_regressor.predict(X)
    fig, axs = plt.subplots(2,1,figsize= (8,10), sharex=True)
    axs[0].plot(measured['ts'],measured['Mass_change'],linewidth=10,label = 'Flow_rate = '+key)
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_change'],label = 'Flow_rate = '+key+' m='+str(slope_dict_1[key][0]))
    axs[0].plot(X,y_hat, color='red')
    axs[1].scatter(measured['ts'],measured['Mass_derivative_smooth'],linewidth=5, color='blue')
    axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],color='orange')
    axs[0].legend()
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass [g]')
    axs[1].set_ylabel('dM/dt [g/s]')
    axs[0].set_xbound(1)
    axs[1].set_xbound(1)
    fig.tight_layout()
    plt.savefig(key+'.png')

# %%
for key in files_dict:
     fig, axs = plt.subplots()
     axs.plot(files_dict[key]['ts'],files_dict[key]['Mass_derivative_smooth'].where(files_dict[key]['Mass_derivative_smooth'].rolling(3).mean()<-1.8),linewidth=10,label = 'Flow_rate = '+key)
     axs.plot(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],label = 'Flow_rate = '+key)
     axs.legend()
# %%Batch Analisis of aspiration curves stoppped by derivative

slope_dict_1 = {}
for key in files_dict:
    counter =0 
    df = pd.DataFrame(columns =files_dict[key].columns)
    for index in range(files_dict[key].shape[0]):
        df = df.append(files_dict[key].iloc[index])
        if df['ts'].iloc[index]>15:
            condition=df['Mass_analysis_derivative_smooth'].rolling(30).mean().iloc[index]
            if condition>-0.05:
                break
    X = files_dict[key].where(files_dict[key]['ts']>15).dropna().loc[:, 'ts'].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = files_dict[key].where(files_dict[key]['ts']>15).dropna().loc[:, 'Mass'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)
    slope_dict_1[key]=linear_regressor.coef_
    y_hat  = linear_regressor.predict(X)

    fig, axs = plt.subplots(2,1,figsize= (8,10), sharex=True)
    axs[0].plot(df['ts'],df['Mass_analysis_smooth'],linewidth=8,label = 'Flow_rate = '+key)
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],label = 'Flow_rate = '+key+' m='+str(slope_dict_1[key][0]))
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],linewidth=8,label = 'Flow_rate = '+key, color = 'green')
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass'],label = 'Flow_rate = '+key, color= 'orange')
    axs[0].plot(X,y_hat, color='red')
    axs[0].legend()
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass [g]')
    axs[1].set_ylabel('dM/dt [g/s]')
    axs[0].set_xbound(5)
    max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].max()+100
    min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].min()-100
    axs[0].set_ybound(min_y,max_y)

    
    axs[1].scatter(df['ts'],df['Mass_analysis_derivative_smooth'],linewidth=10, color='blue')
    axs[1].scatter(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],linewidth=5, color='green')
    axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass_analysis_derivative_smooth'],color='orange')
    max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].max()*1.1
    min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].min()*1.1
    axs[1].set_xbound(5)
    axs[1].set_ybound(min_y,max_y)

    fig.tight_layout()
    #plt.savefig(folder_name+r'/'+key+'.png')





# %%
df = pd.read_csv(r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration\test1.csv')

df['ts'] = df['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
df['ts']=df['ts']-df['ts'][0]
df['Mass_change']=df['Mass']-df['Mass'][0]
df['Mass_smooth']= signal.savgol_filter(df['Mass_change'],5,1)
df['Mass_derivative_smooth']=df['Mass_smooth'].diff()
df.plot(x='ts', y='Mass_derivative_smooth')
# %%
