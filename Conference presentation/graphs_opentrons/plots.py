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



# %%Plot exp3

bar_df_o_exp3 = pd.read_csv('summary_bar_chart_o_exp3.csv')

bar_df_o_exp3['Viscosity']=bar_df_o_exp3['Viscosity'].astype('str')

bar_df_s_exp3 = pd.read_csv(r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Conferences\Conference\Presentation\graphs_sartorious\summary_bar_chart_exp3.csv')

bar_df_s_exp3['Viscosity']=bar_df_s_exp3['Viscosity'].astype('str')

x = np.arange(0,8,2)
width = 0.4

# fig, axs = plt.subplots(figsize=(7.5,5.625))
# axs.bar(x,height=bar_df_exp3.iloc[:,3].abs(),width=width,label = '1000 \mu L', color= 'red')
# axs.bar(x+width,height=bar_df_exp3.iloc[:,8].abs(),width=width,label = '500 \mu L', color= 'green')
# axs.bar(x+2*width,height=bar_df_exp3.iloc[:,13].abs(),width=width,label = '300 \mu L', color= 'black')

# plt.xticks(x + width / 2, ('204 cp', '505 cp', '817 cp','1275 cp'))
x = np.arange(4)
for column in bar_df_o_exp3.iloc[:,[3,8,13]].columns:
    fig, axs = plt.subplots(figsize=(7.5,5.625))
    axs.bar(x=x,height=bar_df_o_exp3[column].abs(),width=width,label = 'ML Driven', color= 'orangered')
    plt.xticks(x + width / 2, ('204 cp', '505 cp', '817 cp','1275 cp'))
    axs.set_xlabel('Viscosity Standards')
    axs.set_ylabel(column)
    fig.tight_layout()
    # fig.savefig('Exp3_'+column+'.png')


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
