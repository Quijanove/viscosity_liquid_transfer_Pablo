#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:30:22 2023

@author: riko i made and Quijanove

The following script will try to suggest the optimize parameters to reach 
desired transfer mass

user input:  
    - 'aspiration_rate', 
    - 'dispense_rate', 
    - 'delay_aspirate', 
    - 'delay_dispense',
    
suggestion: ['volume','aspiration_rate', 
         'dispense_rate', 'delay_aspirate', 
         'delay_dispense'] % error
training data: 817

version 2: add blow out rate

version 3:
        - blow out rate either none or range of value
        - add init-value
        
version 3a:
    - without blowout

It loads a dataframe to input the experimental measure error of the suggested parameters, and allows for iteration of the experimental calibration
of  a target liquid. 
"""
#%%
script_ver = 'version 3a: without blowout'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
%matplotlib qt

import sklearn
from sklearn.preprocessing import StandardScaler


from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

class Dispense:
    
    df = None # prior's dataset
    features = None
    target = None
    model = None # surogate model, either lin = linear, or 'gpr' = gaussian proc
    density = None
    
    asp_max = 25 # aspiration_rate maximum
    asp_min = 20 # aspriation_rate minimum
    
    dsp_max = 13 # dispense_rate maximum
    dsp_min = 8  # dispense_rate minimum
    
    asp_delay_max = 5
    asp_delay_min = 0
    
    dsp_delay_max = 5
    dsp_delay_min = 0
    
    blowout_rate_min = 0
    blowout_rate_max = 10
    
    blowout_delay_min = 0
    blowout_delay_max = 10
    
    vol_min = 100 # micro liter
    vol_max = 1000 # micro liter
    
    def __init__(self, name = 'Unknown'):
        self.name = name
    
    def calibrate(self, volume = list(np.linspace(100,1000,10)), model_kind='gpr'):
        '''
        function to use to calibrate, to find the aspiration and dispense rate
        return: asp_rate, disp_rate
        
        generate surogate function, 
        run gp_minimize to find the next suggestion, with mass constraints
        
        volume_list: insert volume as a list
        '''
        
        if type(volume) != list: volume=[volume]
        
        
        
        from warnings import filterwarnings 
        filterwarnings("ignore")
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.df[self.features])
        self.y_train = np.asarray(self.df[self.target])
        
        self.fit(model_kind)
        
        
        self.space = [Categorical(volume, name='volume'),
                      
                      Real(self.asp_min, self.asp_max, name='aspiration_rate'),
                      Real(self.dsp_min, self.asp_max, name='dispense_rate'),
                      Real(self.asp_delay_min, self.asp_delay_max, name='delay_aspirate'),
                      Real(self.dsp_delay_min, self.dsp_delay_max, name='delay_dispense'),
                      # Categorical([0, 1], name='blowout_state'),
                      # Real(self.blowout_rate_min, self.blowout_rate_max, name='blow_out_rate'),
                      # Real(self.blowout_delay_min, self.blowout_delay_max, name='delay_blow_out')
                      ]
        
        
        
        @use_named_args(self.space)
        def obj_func(**input_array):
            dx = pd.DataFrame()
            for key in input_array.keys():
                
                dx.loc[0,key] = input_array[key] 
            
            # dx.loc[dx['blowout_state'] == 0, ['blow_out_rate', 'delay_blow_out']] = 0

            
            X = self.scaler.transform(dx)
            
            
            pred = self.model.predict(X)
            
            ## scalarization:
            out = pred.item()
            #deleted from above: /(1/input_array['aspiration_rate'] + 1/input_array['dispense_rate'])
            
            return out #pred.item()
        
        #x0 = [list(x) for x in list(np.asarray(self.df[self.features]))]
        #y0 = list(self.df[self.target])
        
        self.res = gp_minimize(obj_func, 
                          self.space, 
                          n_calls=60, 
                          kappa = 1.0, # default 1.95 balanced between exploitation vs exploration
                          acq_func = 'EI',
                          #x0 = x0,
                          #y0 = y0, 
                          random_state=123
                          )
                          
        self.out_df = pd.DataFrame(data=self.res.x_iters)
        self.out_df.columns = [n.name for n in self.space]#self.features
        
        
        # self.out_df.loc[self.out_df['blowout_state'] == 0, ['blow_out_rate', 'delay_blow_out']] =0
            
        
        
        self.out_df['%error'] = self.model.predict(self.scaler.transform(self.out_df[self.features]))
        self.out_df['abs-err'] = abs(self.out_df['%error'])
        self.out_df['oo'] = self.out_df['volume']/self.out_df['aspiration_rate'] \
                            + self.out_df['volume']/self.out_df['dispense_rate'] \
                                + self.out_df['delay_aspirate'] + self.out_df['delay_dispense']
                                
        # Filtering
        
        self.out_df.sort_values(by='abs-err', inplace=True) # sort based on error
        self.out_df.reset_index(inplace=True, drop=True)
        
        self.out_df2 = self.out_df.iloc[:5,:].copy()
        
        # self.out_df2.sort_values(by='oo', ascending=True, inplace=True) ## sort based on time
        # self.out_df2.reset_index(inplace=True, drop=True)
        
        
        print(f'\n {script_ver}\n Next Run:')
        
        for col in list(self.out_df2)[:-1]:
            print('{:>15}\t: {:.1f}'.format(col, self.out_df2.loc[0,col]))
        #return out_df
    
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
            
    
    

# %%


liq = Dispense()

#Please enter name, density, csv with calibration data for training and model name
liq.name = 'Viscosity_std_398'
liq.density = 0.8672
file_name = 'Std_calibrations/{}.csv'.format(liq.name)
model = 'lin'
training_set_list = ['full', 'half','4','1']
training_set = training_set_list[3]
features_list = ['wo_bo', 'wbo']
feature_selection = features_list[0]



#Dont change 
df = pd.read_csv(file_name)
df['blow_out_state'] = np.zeros(df['blow_out_rate'].shape)
df['blow_out_state']= df['blow_out_state'].where(df['blow_out_rate']>0).fillna(0)
df['blow_out_state']= df['blow_out_state'].where(df['blow_out_rate']==0).fillna(1)
features = [['volume','aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense'],['volume','aspiration_rate', 'dispense_rate', 'blow_out_rate',  'delay_aspirate', 'delay_dispense','delay_blow_out']]
target='%error'

if feature_selection == 'wo_bo':
    liq.features = features[0]
else:
    liq.features = features[1]

liq.target = target
if training_set == 'full':
    df = df.iloc[:int(df.shape[0]/2+1)]
elif training_set == '4':
    df = df.iloc[:int(df.shape[0]/4+1)]
elif training_set == '1':
    df = df.iloc[:1]


#No need to change
liq.asp_max = df['aspiration_rate'][0] * 1.2 
liq.asp_min = df['aspiration_rate'][0] * 0.2
liq.dsp_max = df['dispense_rate'][0] * 1.2 
liq.dsp_min = df['dispense_rate'][0] * 0.1
liq.blowout_rate_max = liq.asp_max
liq.blowout_rate_min = 0

liq.asp_delay_max = 6 
liq.asp_delay_min = 0
liq.dsp_delay_max = 6 
liq.dsp_delay_min = 0
liq.blowout_delay_max = 6
liq.blowout_delay_min = 0




#%%
folder = './Opentrons_experiments'
today = date.today().strftime("%Y-%m-%d")
subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
if liq.name.split('.')[0] not in subfolders:
    os.mkdir(folder+'/'+liq.name.split('.')[0])
    os.mkdir(folder+'/'+liq.name.split('.')[0]+'/lin')
    os.mkdir(folder+'/'+liq.name.split('.')[0]+'/lin/df2')
    os.mkdir(folder+'/'+liq.name.split('.')[0]+'/gpr')
    os.mkdir(folder+'/'+liq.name.split('.')[0]+'/gpr/df2')
counter =1 



#%%Run for each iteration do not change
liq.df = df
liq.calibrate(1000) ## input volume, when blank it will chose a value between 100 - 1000 uL, 

df = df.append(liq.out_df2.iloc[0,0:-2],ignore_index=True)
df.iloc[-1,0:5] = df.iloc[0,0:5]
df.loc[:,'touch_tip_aspirate'].iloc[-1] = df.loc[:,'touch_tip_aspirate'].iloc[0]
df.loc[:,'touch_tip_dispense'].iloc[-1] = df.loc[:,'touch_tip_dispense'].iloc[0]
df.loc[:,'blow_out_state'].iloc[-1] = 1
df.loc[:,'blow_out_rate'].iloc[-1] = 0
df.loc[:,'delay_blow_out'].iloc[-1] = 0

df['m_expected'].iloc[-1]=df['volume'].iloc[-1]/1000 * liq.density

counter +=1 
liq.out_df2.to_csv(folder+'/'+liq.name.split('.')[0]+'/'+model+'/'+'df2/'+training_set+'_'+ date.today().strftime("%Y-%m-%d")+'_'+datetime.now().strftime("%H-%M")+'.csv', index = False)

#%%
df['m_measured'].iloc[-1]= 0.8548         

df['time'].iloc[-1]= 63.0953

df[r'%error'].iloc[-1]= (df['m_measured'].iloc[-1]- df['m_expected'].iloc[-1])/df['m_expected'].iloc[-1] *100
df.to_csv('current_experiment.csv', index=False)
df


df = pd.read_csv('current_experiment.csv')

#%%
df.to_csv(folder+'/'+liq.name.split('.')[0]+'/'+model+'/'+training_set+'_'+feature_selection+'_'+today+'_'+datetime.now().strftime("%H-%M")+'.csv', index = False)

# %%
