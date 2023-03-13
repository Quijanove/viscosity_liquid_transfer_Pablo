# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:30:22 2023

@author: rikoim

The following script will try to suggest the optimize parameters to reach 
desired transfer mass

user input: 'm_expected': the desired mass
optimized parameters : 
    - 'aspiration_rate', 
    - 'dispense_rate', 
    - 'delay_aspirate', 
    - 'delay_dispense',
    
suggestion: ['m_expected','aspiration_rate', 
         'dispense_rate', 'delay_aspirate', 
         'delay_dispense'] % error
training data: 817
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sklearn
#from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split, LeaveOneOut

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

class Squirt:
    
    df = None # prior's dataset
    features = None
    target = None
    model = None # surogate model, either lin = linear, or 'gpr' = gaussian proc
    
    asp_max = 25 # aspiration_rate maximum
    asp_min = 20 # aspriation_rate minimum
    
    dsp_max = 13 # dispense_rate maximum
    dsp_min = 8  # dispense_rate minimum
    
    asp_delay_max = 5
    asp_delay_min = 0
    
    dsp_delay_max = 5
    dsp_delay_min = 0
    
    def __init__(self, name = 'Unknown'):
        self.name = name
    
    def calibrate(self, mass, pct=2, model_kind='lin'):
        '''
        function to use to calibrate, to find the aspiration and dispense rate
        return: asp_rate, disp_rate
        
        generate surogate function, 
        run gp_minimize to find the next suggestion, with mass constraints
        '''
        from warnings import filterwarnings 
        filterwarnings("ignore")
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.df[self.features])
        self.y_train = np.asarray(self.df[self.target])
        
        self.fit(model_kind)
        
        
        self.space = [Categorical([mass], name='m_expected'),
                      Real(self.asp_min, self.asp_max, name='aspiration_rate'),
                      Real(self.dsp_min, self.asp_max, name='dispense_rate'),
                      Real(self.asp_delay_min, self.asp_delay_max, name='delay_aspirate'),
                      Real(self.dsp_delay_min, self.dsp_delay_max, name='delay_dispense'),
            ]
        @use_named_args(self.space)
        def obj_func(**input_array):
            dx = pd.DataFrame()
            for key in input_array.keys():
                
                dx.loc[0,key] = input_array[key] 
                
            # print(dx)
            #input_array = np.asarray(dx)
            
            X = self.scaler.transform(dx)
            y = input_array['m_expected']
            
            pred = self.model.predict(X)
            
            
            abs_error = abs(y - pred)/y*100
            
            return abs_error.item()
        
        self.res = gp_minimize(obj_func, 
                          self.space, 
                          n_calls=50, 
                          #x0 = np.asarray(self.df[self.features]),
                          #y0 = self.y_train.reshape(-1,1)
                          )
        self.Xi = self.res.x
        self.fun = self.res.fun
        self.Xi_dict = {}
        
        for i, k in enumerate(self.space):
            self.Xi_dict[k.name] = self.Xi[i]
            
        self.Xi_dict['%error'] = self.fun
        
        print('\nNext Run:')
        
        for k in self.Xi_dict.keys():
            print('{:>15}\t: {:.3f}'.format(k, self.Xi_dict[k]))
        return self.Xi, self.fun
    
    def fit(self, kind='gpr'):
        '''
        lin: linear
        gpr: gpr
        '''
        
        if kind == 'gpr':
            matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

            self.model = GaussianProcessRegressor(kernel=matern_tunable, 
                                    n_restarts_optimizer=10, 
                                    alpha=0.5, 
                                      normalize_y=True)
            self.model.fit(self.X_train, self.y_train)
        
        else:
            self.model= sklearn.linear_model.LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            
    
    
    def transfer(self, mass):
        '''
        function to use to transfer liquid in production
        input mass required
        
        return: asp_rate, disp_rate
        '''
        pass
    
    

 
if __name__ == '__main__':   
    df = pd.read_csv('practice_data.csv')
    
    
    features = ['m_expected','aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense']
    target='m_measured'
    # %%
    
    liq = Squirt()
    liq.name = 'Not-unknown'
    liq.features = features
    liq.df = df
    liq.target = target
    
    liq.calibrate(.8) ## input desired mass, 

