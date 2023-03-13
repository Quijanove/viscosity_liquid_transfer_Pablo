#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from  sklearn.preprocessing import StandardScaler
import sklearn.linear_model as linear_model
from sklearn.model_selection import cross_validate
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

#%%
file_name = r'Std_calibrations/Viscosity_std_398.csv'

df = pd.read_csv(file_name)
features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr']
for model_name in model_list:
    if model_name == 'lin':
        model = linear_model.LinearRegression()

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, 
                                        n_restarts_optimizer=10, 
                                        alpha=0.5, 
                                        normalize_y=True)

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled = df_scaled.sample(frac=1,random_state=42)

    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    X = df_scaled[features]
    y = df_scaled[target]
    loo = LeaveOneOut()

    scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
    print(abs(scores['test_score']).mean())

#%%




# %%
