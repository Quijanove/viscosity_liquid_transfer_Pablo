#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%% Bar chart

bar_df_exp2 = pd.read_csv('summary_bar_chart_exp3.csv')

bar_df_exp2['Viscosity']=bar_df_exp2['Viscosity'].astype('str')

# %%Plot exp3
x = np.arange(4)
width = 0.4

for column in bar_df.columns[1:-1]:
    fig, axs = plt.subplots(figsize=(7.5,5.625))
    bar_df_man = bar_df_exp2.where(bar_df_exp2.Driver=='Human').dropna()
    bar_df_ML= bar_df_exp2.where(bar_df_exp2.Driver=='ML').dropna()
    axs.bar(x=x,height=bar_df_man[column].abs(), width=width, label = 'Human Driven', color = 'cornflowerblue')
    axs.bar(x=x+width,height=bar_df_ML[column].abs(),width=width,label = 'ML Driven', color= 'orangered')
    plt.xticks(x + width / 2, ('204 cp', '505 cp', '817 cp','1275 cp'))
    axs.set_xlabel('Viscosity Standards')
    axs.set_ylabel(column)
    fig.tight_layout()
    fig.savefig('Exp3_'+column+'.png')


# %% Plot experiment 2

bar_df_exp2 = pd.read_csv('summary_bar_chart_exp2.csv')

bar_df_exp2['Viscosity']=bar_df_exp2['Viscosity'].astype('str')
bar_df_man = bar_df_exp2.where(bar_df_exp2.Driver=='Human').dropna()
bar_df_ML= bar_df_exp2.where(bar_df_exp2.Driver=='ML').dropna()
x = np.arange(4)
width = 0.4

for column in bar_df_exp2.columns[1:-3]:
    fig, axs = plt.subplots(figsize=(7.5,5.625))
    axs.bar(x=x,height=bar_df_man[column].abs(), width=width, label = 'Human Driven', color = 'cornflowerblue')
    axs.bar(x=x+width,height=bar_df_ML[column].abs(),width=width,label = 'ML Driven', color= 'orangered')
    plt.xticks(x + width / 2, ('204 cp', '505 cp', '817 cp','1275 cp'))
    axs.set_xlabel('Viscosity Standards')
    axs.set_ylabel(column)
    fig.tight_layout()
    # fig.savefig('Exp3_'+column+'.png')
x = np.arange(0,10,2.5)
for column in bar_df_exp2.columns[-3:-1]:
    fig, axs = plt.subplots(figsize=(7.5,5.625))
    axs.bar(x=x,height=bar_df_man[column].astype('float'), width=width, label = 'Human Driven', color = 'cornflowerblue')
    for i,row in enumerate(bar_df_ML[column]):
        values_per_volume = row.replace('[','').replace(']','').split(',')
        axs.bar(x=x[i:i+1].repeat(3)+[width, width*2, width*3],height=np.array(values_per_volume,dtype=int),width=width,label = 'ML Driven', color= 'orangered', edgecolor=['red','green','black'],linewidth=6)
    plt.xticks(x + width / 2, ('204 cp', '505 cp', '817 cp','1275 cp'))
    axs.set_xlabel('Viscosity Standards')
    axs.set_ylabel(column)
    axs.set_ybound(lower=0)
    axs.autoscale(False)
    fig.tight_layout()
    # fig.savefig('Exp3_'+column+'.png')
# %%
with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_ylim([-2, 2])

    # data = np.ones(100)
    # data[70:] -= np.arange(30)

    # ax.annotate(
    #     'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
    #     xy=(70, 1), arrowpr|ops=dict(arrowstyle='->'), xytext=(15, -10))

    ax.plot(-x,data_sig)
    ax.plot(x,data_gauss)

    ax.set_xlabel('time')
    ax.set_ylabel('my overall health')
    fig.text(
        0.5, 0.05,
        '"Stove Ownership" from xkcd by Randall Munroe',
        ha='center')    

# %%
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def gauss(x,a,b,c):
    y = a*np.exp(-(x-b)**2/(2*c**2))

    return y 

x = np.arange(-200,150,1)
data_sig = sigmoid(x, 10,0,0.08,0)
data_gauss = gauss(x,10,10,100)+10


# %%
plot