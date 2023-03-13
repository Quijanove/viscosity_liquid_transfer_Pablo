"""
Created on Tue Feb  7 13:30:22 2023

@author: pabloquijanove

The following script can be used to process csv files saved from 
Opentrons experiments into csv files that can be read by Dispense class
"""
#%%
import pandas as pd
import os
#%%
for i in os.listdir(): 
    if 'csv' in i:     
        df = pd.read_csv(i)
        if 'm_expected' not in df.columns:
            df['m_measured'] = df['m']
            df['m_expected'] = df['density']*df['volume']/1000
            df= df.drop(['mi','mf','Transfer_Observation', 'Comment','density','m'], axis=1)
            df.insert(1, 'Viscosity 10 s-1',[df['liquid'][0].split('_')[-1]]*df.shape[0])
            df.insert(1, 'Viscosity 100 s-1',[df['liquid'][0].split('_')[-1]]*df.shape[0])
            df.insert(1, 'Viscosity 900 s-1',[df['liquid'][0].split('_')[-1]]*df.shape[0])
            df.to_csv(i, index= False)
        

# %%
