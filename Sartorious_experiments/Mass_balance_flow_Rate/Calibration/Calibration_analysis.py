#%%
import pandas as pd
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
%matplotlib qt
import os
import regex as re
from sklearn.linear_model import LinearRegression
import matplotlib.animation as animation
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

# controls default text sizes
plt.rc('lines',markersize=8)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 

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
    files_dict[key]['Mass_analysis_smooth']= signal.savgol_filter(files_dict[key]['Mass'],91,89)
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
    # df = pd.DataFrame(columns =files_dict[key].columns)
    # for index in range(files_dict[key].shape[0]):
    #     df = df.append(files_dict[key].iloc[index])
    #     if df['ts'].iloc[index]>75:
    #         condition=df['Mass_analysis_derivative_smooth'].rolling(30).mean().iloc[-1]
    #         if condition>-0.05:
    #             break
    # # X = files_dict[key].where(files_dict[key]['ts']>15).dropna().loc[:, 'ts'].values.reshape(-1, 1)  # values converts it into a numpy array
    # Y = files_dict[key].where(files_dict[key]['ts']>15).dropna().loc[:, 'Mass'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    # linear_regressor = LinearRegression()  # create object for the class
    # linear_regressor.fit(X, Y)
    # slope_dict_1[key]=linear_regressor.coef_
    # y_hat  = linear_regressor.predict(X)

    fig, axs = plt.subplots(2,1,figsize= (8,10), sharex=True)
    # axs[0].plot(df['ts'],df['Mass_analysis_smooth'],linewidth=16,label = 'Flow_rate = '+key)
    # axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],label = 'Flow_rate = '+key+' m=')#+str(slope_dict_1[key][0]))
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],linewidth=8,label='Processed data', color = 'green')
    axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass'],label = 'Experimental data', color= 'orange')
    # axs[0].plot(X,y_hat, color='red')
    axs[0].legend()
    axs[1].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass [g]')
    axs[1].set_ylabel('dM/dt [g/s]')
    axs[0].set_xbound(5)
    max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].max()+100
    min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].min()-100
    axs[0].set_ybound(min_y,max_y)

    
    # axs[1].scatter(df['ts'],df['Mass_analysis_derivative_smooth'],linewidth=10, color='blue')
    axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass'].diff(),color='orange')
    axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],linewidth=5, color='green')
    max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].max()*1.1
    min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].min()*1.1
    axs[1].set_xbound(5)
    axs[1].set_ybound(min_y,max_y)
    axs[0].yaxis.set_tick_params(labelleft=False)
    axs[1].yaxis.set_tick_params(labelleft=False)

    fig.tight_layout()
    # counter+=1
    # if counter ==1:
    #     break
    # plt.savefig(folder_name+r'/'+key+'.png')


# %%Batch Analisis of aspiration curves stoppped by derivative
key =  '2023-03-10_16-19_Viscosity_std_1275_265_csv'
fig, axs = plt.subplots(2,1,figsize= (8,10), sharex=True)
axs[0].yaxis.set_tick_params(labelleft=False)
axs[1].plot(np.arange(0,90,1),[-0.05]*90)

line, = axs[0].plot(files_dict[key]['ts'][:], files_dict[key]['Mass_smooth'][:],linewidth=5)

line2, = axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],linewidth=5)


def animate(i):
    line.set_ydata(files_dict[key]['Mass_smooth'][:350+i])
    line.set_xdata(files_dict[key]['ts'][:350+i])
    line2.set_ydata(files_dict[key]['Mass_derivative_smooth'][:350+i])
    line2.set_xdata(files_dict[key]['ts'][:350+i])
    return line,line2,
ani = animation.FuncAnimation(
    fig, animate, frames= 350,interval=20, blit=True)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True)
plt.show()
#%%
# axs[0].plot(df['ts'],df['Mass_analysis_smooth'],linewidth=16,label = 'Flow_rate = '+key)
# axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],label = 'Flow_rate = '+key+' m=')#+str(slope_dict_1[key][0]))
axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass_smooth'],linewidth=8,label='Processed data', color = 'green')
axs[0].plot(files_dict[key]['ts'], files_dict[key]['Mass'],label = 'Experimental data', color= 'orange')
# axs[0].plot(X,y_hat, color='red')
axs[0].legend()
axs[1].set_xlabel('Time [s]')
axs[0].set_ylabel('Mass [g]')
axs[1].set_ylabel('dM/dt [g/s]')
axs[0].set_xbound(5)
max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].max()+100
min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna()['Mass'].min()-100
axs[0].set_ybound(min_y,max_y)


# axs[1].scatter(df['ts'],df['Mass_analysis_derivative_smooth'],linewidth=10, color='blue')
axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass'].diff(),color='orange')
axs[1].plot(files_dict[key]['ts'], files_dict[key]['Mass_derivative_smooth'],linewidth=5, color='green')
max_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].max()*1.1
min_y = files_dict[key].where(files_dict[key]['ts']>5).dropna(how='all')['Mass_analysis_derivative_smooth'].min()*1.1
axs[1].set_xbound(5)
axs[1].set_ybound(min_y,max_y)
axs[0].yaxis.set_tick_params(labelleft=False)
axs[1].yaxis.set_tick_params(labelleft=False)

fig.tight_layout()
# counter+=1
# if counter ==1:
#     break
# plt.savefig(folder_name+r'/'+key+'.png')


# %%
df = pd.read_csv(r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration\test1.csv')

df['ts'] = df['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
df['ts']=df['ts']-df['ts'][0]
df['Mass_change']=df['Mass']-df['Mass'][0]
df['Mass_smooth']= signal.savgol_filter(df['Mass_change'],5,1)
df['Mass_derivative_smooth']=df['Mass_smooth'].diff()
df.plot(x='ts', y='Mass_derivative_smooth')
# %%
from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

file = '2023-03-10_16-57_Viscosity_std_505_265_csv'

xdata = files_dict[file].where(files_dict[file]['ts']>15).dropna()['ts']
ydata = files_dict[file].where(files_dict[file]['ts']>15).dropna()['Mass']

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0)

yfit = sigmoid(xdata,popt[0],popt[1],popt[2],popt[3])

fig,axs = plt.subplots(2)
axs[0].plot(xdata,ydata,color = 'red')
axs[0].plot(xdata,yfit,color = 'blue')
axs[1].plot(xdata,yfit.diff(),color = 'green')

# %%
from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

for file in files_dict.keys():

    xdata = files_dict[file].where(files_dict[file]['ts']>15).dropna()['ts']
    ydata = files_dict[file].where(files_dict[file]['ts']>15).dropna()['Mass']

    p0 = [max(ydata)+30, np.median(xdata),1,min(ydata)] # this is an mandatory initial guess
    print(p0)

    popt, pcov = curve_fit(sigmoid, xdata, ydata,p0)

    yfit = sigmoid(xdata,popt[0],popt[1],popt[2],popt[3])
    
    print('k='+str(popt[2]))

    fig,axs = plt.subplots(2)
    axs[0].plot(xdata,ydata,color = 'red', label='Experimental data'+file)
    axs[0].plot(xdata,yfit,color = 'blue', label= 'Sigmoid fit')
    axs[0].legend()
    axs[1].plot(xdata,yfit.diff(),color = 'green', label= 'fit derivative')
    axs[1].legend()

    axs[1].set_xlabel('Time [s]')
    axs[0].set_ylabel('Mass [g]')
    axs[1].set_ylabel('dM/dt [g/s]')
    axs[0].yaxis.set_tick_params(labelleft=False)
    axs[1].yaxis.set_tick_params(labelleft=False)

    fig.tight_layout()

    plt.savefig(folder_name+r'/'+file.split('_265')[0][17:]+'_sigmoid.png')


# %%
