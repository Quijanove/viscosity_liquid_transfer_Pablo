#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
%matplotlib qt

import sklearn
from sklearn.preprocessing import StandardScaler

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
#%%
def obj_func(df):
    features = ['volume','aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense']
    dx = df[features]
    scaler = StandardScaler()
    matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)
    model = GaussianProcessRegressor(kernel=matern_tunable, 
                        n_restarts_optimizer=10, 
                        alpha=0.5, 
                            normalize_y=True)
    for index in dx.index:   
        X = scaler.fit_transform(dx.iloc[:index+1])
        pred = model.predict(X)
        
        ## scalarization:
        out = pred.item()/(1/row['aspiration_rate'] + 1/row['dispense_rate'])
        #deleted from above:
        dx['objective function'].iloc[index] = out     
    return dx 


#%%
file_path = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\GitHub\viscosity_liquid_transfer_Pablo\Opentrons_experiments\Viscosity_std_398\Data_collected_from_opentrons\Viscosity_std_398_ML_training_1_wo_bo_lin_divide_unorderedT_distributed.csv'
df = pd.read_csv(file_path)

obj_func(df)




# %%
