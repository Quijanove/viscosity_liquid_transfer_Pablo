"""
author: Quijanove
This script is to plot and analyze the data generated while testing different ML models to drive
the optimization of liquid transfer parameters of viscous liquids using automated pipetting
robots.
"""
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os #os is a package that let's you get paths of files in different folders 
# %matplotlib qt
#%%
# This cell will load into a dictioanry all the csv files containing the transfer data 
# from the files generated in the Opentron's jupyter notebook and in the visc_vx_measurement.py
# given a specific viscous liquid name and a model

liquid_name = 'Viscosity_std_1275'
version = 'wo_bo'
data_train = '1'

viscous_liquid_cal_folder = '../Std_calibrations'
viscous_liquid_exp_folder = '../Opentrons_experiments/'+liquid_name + '/Data_collected_from_opentrons' 
file_dict  = {liquid_name:{}}


for name in os.listdir(viscous_liquid_cal_folder):
    if  liquid_name in name:
        file_dict[liquid_name]['Cal']= {name[:-4] : pd.read_csv(viscous_liquid_cal_folder+'/'+name)}

file_dict[liquid_name]['Experiment'] = {}
for name in os.listdir(viscous_liquid_exp_folder):
    if '.csv' in name:
        if version in name and data_train in name.split('training')[1]:
            file_dict[liquid_name]['Experiment'].update({name[:-4]: pd.read_csv(viscous_liquid_exp_folder+'/'+name)})
    

#%%
df_cal = pd.read_csv(viscous_liquid_cal_folder+'/'+liquid_name+'.csv')

if data_train == 'full': 

    for experiment in file_dict[liquid_name]['Experiment']:
    
        if 'lin' in experiment:
            model = 'Linear Regression'
        elif 'gpr' in experiment:
            model = 'Gaussian Process Regression'
        
        if 'none' in experiment:
            penalization = 'no'

        elif 'multiply' in experiment:
            penalization = 'slow transfer'

        elif 'divide' in experiment:
            penalization = 'fast transfer'

        df_experiment = file_dict[liquid_name]['Experiment'][experiment]
        fig,axs = plt.subplots(2,1)

        df_cal_1000 = df_cal.where(df_cal.volume==1000).dropna(how='all')
        df_cal_500 = df_cal.where(df_cal.volume==500).dropna(how='all')
        df_cal_300 = df_cal.where(df_cal.volume==300).dropna(how='all')

        df_auto_1000 = df_experiment.where(df_experiment.volume==1000).dropna(how='all') #df_auto dataframe that holds the data suggested by ML program
        df_auto_500 = df_experiment.where(df_experiment.volume==500).dropna(how='all')
        df_auto_300 = df_experiment.where(df_experiment.volume==300).dropna(how='all')

        axs[0].scatter(df_cal_1000.index.to_series()+1,df_cal_1000['%error'], marker= 'x', label = '1000', c = 'red')
        axs[0].scatter(df_cal_500.index.to_series()+1,df_cal_500['%error'], marker= 'x', label = '500', c = 'green')
        axs[0].scatter(df_cal_300.index.to_series()+1,df_cal_300['%error'], marker= 'x', label = '300', c = 'grey')
        axs[0].plot(df_cal.index.to_series()+1, df_cal['%error'],label = 'Human driven' )

        axs[0].scatter(df_auto_1000.index.to_series()+df_cal.index.to_series().iloc[-1]+1,df_auto_1000['%error'], marker= 'x', c = 'red')
        axs[0].scatter(df_auto_500.index.to_series()+df_cal.index.to_series().iloc[-1]+1,df_auto_500['%error'], marker= 'x', c = 'green')
        axs[0].scatter(df_auto_300.index.to_series()+df_cal.index.to_series().iloc[-1]+1,df_auto_300['%error'], marker= 'x',  c = 'grey')
        axs[0].plot(df_experiment.index.to_series()+df_cal.index.to_series().iloc[-1]+1, df_experiment['%error'],label = 'ML driven' )

        axs[0].plot([df_cal.index.to_series().iloc[-1]+1]*2,  np.append(df_cal['%error'].iloc[-1],df_experiment['%error'].iloc[0]))



        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Error [%]')


        axs[1].scatter(df_cal_1000.index.to_series()+1,df_cal_1000['time'], marker= 'x', c = 'red')
        axs[1].plot(df_cal_1000.index.to_series()+1,df_cal_1000['time'])
        
        axs[1].scatter(df_experiment.index.to_series()+1,df_experiment['time'],marker= 'x',c='red')
        axs[1].plot(df_experiment.index.to_series()+1,df_experiment['time'] )

        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Time')      

        fig.suptitle('{} model with {}  penalization, \n trained with {} data set'.format(model,penalization,data_train))
        fig.legend(loc=7)
        fig.tight_layout()
        fig.savefig(viscous_liquid_exp_folder+r'/'+experiment+'.png')


if data_train == '1':       

    for experiment in file_dict[liquid_name]['Experiment']:

        if 'lin' in experiment:
            model = 'Linear Regression'
        elif 'gpr' in experiment:
            model = 'Gaussian Process Regression'
        
        if 'none' in experiment:
            penalization = 'no'

        elif 'multiply' in experiment:
            penalization = 'slow transfer'

        elif 'divide' in experiment:
            penalization = 'fast transfer'

        df_experiment = file_dict[liquid_name]['Experiment'][experiment]
        fig,axs = plt.subplots(2,1)

        df_cal_1000 = df_cal.where(df_cal.volume==1000).dropna(how='all')
        df_cal_500 = df_cal.where(df_cal.volume==500).dropna(how='all')
        df_cal_300 = df_cal.where(df_cal.volume==300).dropna(how='all')

        df_auto_1000 = df_experiment.where(df_experiment.volume==1000).dropna(how='all') #df_auto dataframe that holds the data suggested by ML program
        df_auto_500 = df_experiment.where(df_experiment.volume==500).dropna(how='all')
        df_auto_300 = df_experiment.where(df_experiment.volume==300).dropna(how='all')

        axs[0].scatter(df_cal_1000.index.to_series()+1,df_cal_1000['%error'], marker= 'x', label = '1000', c = 'red')
        axs[0].scatter(df_cal_500.index.to_series()+1,df_cal_500['%error'], marker= 'x', label = '500', c = 'green')
        axs[0].scatter(df_cal_300.index.to_series()+1,df_cal_300['%error'], marker= 'x', label = '300', c = 'grey')
        axs[0].plot(df_cal.index.to_series()+1, df_cal['%error'],label = 'Human driven' )

        axs[0].scatter(df_auto_1000.index.to_series()+2,df_auto_1000['%error'], marker= 'x', c = 'red')
        axs[0].scatter(df_auto_500.index.to_series()+2,df_auto_500['%error'], marker= 'x', c = 'green')
        axs[0].scatter(df_auto_300.index.to_series()+2,df_auto_300['%error'], marker= 'x',  c = 'grey')
        axs[0].plot(df_experiment.index.to_series()+2, df_experiment['%error'],label = 'ML driven' )
        axs[0].plot(df_cal.index.to_series().iloc[0:2]+1,  np.append(df_cal['%error'].iloc[0],df_experiment['%error'].iloc[0]),color = '#ff7f0e')




        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Error [%]')
        # axs[0].legend(loc='lower right')

    
        axs[1].scatter(df_cal_1000.index.to_series()+1,df_cal_1000['time'], marker= 'x', c = 'red')
        axs[1].plot(df_cal_1000.index.to_series()+1,df_cal_1000['time'])

        axs[1].scatter(df_experiment.index.to_series()+2,df_experiment['time'],marker= 'x',c='red')
        axs[1].plot(df_experiment.index.to_series()+2,df_experiment['time'] )
        axs[1].plot(df_cal.index.to_series().iloc[0:2]+1,  np.append(df_cal['time'].iloc[0],df_experiment['time'].iloc[0]),color = '#ff7f0e')


        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Time')
        # axs[1].legend(loc='lower right')
        

        fig.suptitle('{} model with {}  penalization, \n trained with {} initialization data'.format(model,penalization,data_train))
        fig.legend(loc=7)
        fig.tight_layout()
        fig.savefig(viscous_liquid_exp_folder+r'/'+experiment+'.png')

# %%
