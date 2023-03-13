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
import os 
#%%
liquid_name = 'Viscosity_std_398'
model = 'gpr'
viscous_liquid_folder = r'Opentrons_experiments/'+liquid_name + '/' + model  
file_dict  = {}

for name in os.listdir(viscous_liquid_folder):
    if os.path.isfile(viscous_liquid_folder+'/'+name) == True:
        file_dict[liquid_name]= {name[:-4] : pd.read_csv(viscous_liquid_folder+'/'+name)}

opentrons_record_folder = r'Opentrons_experiments/'+liquid_name + '/' +'Data_collected_from_opentrons'

for name in os.listdir(opentrons_record_folder):
   if model in name:
    file_dict[liquid_name][name[:-4]]= pd.read_csv(opentrons_record_folder+'/'+name)


#%%One file

name_1 = 'full_2023-03-02'
name_2 = 'Viscosity_std_398_ML_training_full_gpr'

df_man = file_dict[liquid_name][name_1]

ml_size = file_dict[liquid_name][name_2].shape[0] 

fig,axs = plt.subplots()

df_man_1000 = df_man[:-ml_size].where(df.volume==1000).dropna()
df_man_500 = df_man[:-ml_size].where(df.volume==500).dropna()
df_man_300 = df_man[:-ml_size].where(df.volume==300).dropna()

df_auto_1000 = df_man[-1-ml_size:].where(df.volume==1000).dropna()
df_auto_500 = df_man[-1-ml_size:].where(df.volume==500).dropna()
df_auto_300 = df_man[-1-ml_size:].where(df.volume==300).dropna()

axs.scatter(df_man_1000.index.to_series()+1,df_man_1000['%error'], marker= 'x', label = '1000', c = 'red')
axs.scatter(df_man_500.index.to_series()+1,df_man_500['%error'], marker= 'x', label = '500', c = 'green')
axs.scatter(df_man_300.index.to_series()+1,df_man_300['%error'], marker= 'x', label = '300', c = 'grey')
axs.plot(df_man[:-ml_size].index.to_series()+1, df_man['%error'][:-ml_size],label = 'Manual' )

axs.scatter(df_auto_1000.index.to_series()+1,df_auto_1000['%error'], marker= 'x', c = 'red')
axs.scatter(df_auto_500.index.to_series()+1,df_auto_500['%error'], marker= 'x', c = 'green')
axs.scatter(df_auto_300.index.to_series()+1,df_auto_300['%error'], marker= 'x',  c = 'grey')
axs.plot(df_man[-1-ml_size:].index.to_series()+1, df_man['%error'][-1-ml_size:],label = 'Automated' )


axs.set_xlabel('Iteration')
axs.set_ylabel('Error [%]')
axs.legend()



# %%

