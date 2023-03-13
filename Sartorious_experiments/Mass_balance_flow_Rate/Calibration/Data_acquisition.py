#%%
"""author: Pablo Quijano 
This python file can be used to record flow rate data using a Sartorious rLine pipette
"""
#%% Import packages
from datetime import datetime, date
import os
import time
import pandas as pd
from scipy import signal
import plotly.express as px 
pd.options.plotting.backend = 'plotly'
from matplotlib import pyplot as plt
%matplotlib qt

from pathlib import Path
import sys
REPO_address = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\GitHub\polylectric'

sys.path.append(REPO_address)

#%% Initialize controlable 

from configs.SynthesisB1 import SETUP, LAYOUT_FILE

from controllably import load_deck      # optional
load_deck(SETUP.setup, LAYOUT_FILE)     # optional

this = SETUP
this.mover.verbose = False
#%% Create variables to control mass balance and pipette 
mover = this.mover
liquid = this.liquid
balance = this.balance
deck = this.setup.deck
tip_rack= deck.get_slot('3')
balance_deck = deck.get_slot('1')

balance.zero(wait=5)

#%%
this.setup.attachTipAt(tip_rack.wells['B1'].top)

#%% This cell can be used to measure the mass change profiles when you aspirate a liquid at different speed rates

folder = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration'
today = date.today().strftime("%Y-%m-%d")
now = datetime.now(tz=None).strftime("%H-%M-%S")
if  not os.path.exists(folder+'\\'+today):
    os.mkdir(folder+'\\'+today)
folder = folder+'\\'+today

volume = 1000


for speed in range(10,270,20):
    filename = folder + '/' +'/'+ today + " " + now + ' ' + str(speed).replace('.','_') + ".csv"
    balance.zero(wait=5)
    balance.clearCache()
    balance.toggleRecord(on=True)
    liquid.aspirate(volume,speed=speed)
    time.sleep(10)
    balance.toggleRecord(on=False)
    balance.buffer_df.to_csv(filename)
    liquid.dispense(volume)
    time.sleep(10)

        
#%%This cell can be used to measure the mass change profiles when you aspirate a liquid at different speed rates.
#The run is automatically stopped when the mass change from the vesel is larger than -990

folder = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration'
today = date.today()
today = today.strftime("%Y-%m-%d")
now = datetime.now(tz=None)
now = now.strftime("%H-%M-%S")
if  not os.path.exists(folder+'\\'+today):
    os.mkdir(folder+'\\'+today)
folder = folder+'\\'+today

volume = 1000

slope_dict = {}

for speed in range(10,280,20):
    filename = folder + '/' +'/'+ today("%Y-%m-%d") + " " + now("%H-%M-%S") + ' ' + str(speed).replace('.','_') + ".csv"
    balance.clearCache()
    balance.toggleRecord(on=True)
    liquid.aspirate(volume,speed=speed)
    while balance.buffer_df['Mass'].iloc[-1] >-990:
        pass
    time.sleep(5)
    balance.toggleRecord(on=False)
    balance.buffer_df.to_csv(filename)
    liquid.dispense(volume)
    fig, axs = plt.subplots()
    measurement=pd.read_csv(filename)
    measurement['ts'] = measurement['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
    data = measurement.where(measurement['Mass'].between(-800,-200)).dropna()
    X = data.loc[:, 'ts'].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.loc[:, 'Mass'].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)
    slope_dict[speed]=linear_regressor.coef_
    y_hat  = linear_regressor.predict(X)
    axs.plot(measurement['ts'],measurement['Mass'],label = 'Flow_rate = '+str(speed)+' m='+str(slope_dict[speed][0]))
    axs.plot(X,y_hat, color='red')
    axs.set_xbound(1)
    axs.legend()
    plt.savefig(filename[:-4]+'.png')
    time.sleep(10)

#%%This cell can be used to measure the mass change profiles when you aspirate a liquid at different speed rates.
#The run is automatically stopped when the mass change derivative is close to 0

folder = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration'
today = date.today()
today = today.strftime("%Y-%m-%d")
now = datetime.now(tz=None)
now = now.strftime("%H-%M-%S")
if  not os.path.exists(folder+'\\'+today):
    os.mkdir(folder+'\\'+today)
folder = folder+'\\'+today

volume = 1000


for speed in range(10,280,20):
    filename = folder + '/' +'/'+ today + " " + now + ' ' + str(speed).replace('.','_') + ".csv"
    balance.zero(wait=5)
    balance.clearCache()
    balance.toggleRecord(on=True)
    liquid.aspirate(volume,speed=speed)
    while True:
        data = balance.buffer_df
        if speed<30:
            data['Mass_smooth']= signal.savgol_filter(data['Mass'],31,1)
        else:
            data['Mass_smooth']= signal.savgol_filter(data['Mass'],5,1)
        data['Mass_derivative_smooth']=data['Mass_smooth'].diff()
        condition=data['Mass_derivative_smooth'].rolling(3).mean().iloc[-1]
        if  condition>-0.8 and condition<0.5:
            break
    time.sleep(5)
    balance.toggleRecord(on=False)
    balance.buffer_df.to_csv(filename)
    liquid.dispense(volume)
    time.sleep(10)

        
#%%This cell can be used to measure the mass change profiles when you aspirate a liquid at different speed rates.
#The run is automatically stopped when the mass change derivative is close to 0
folder = r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration'
today = date.today()
today = today.strftime("%Y-%m-%d")
now = datetime.now(tz=None)
now = now.strftime("%H-%M-%S")
if  not os.path.exists(folder+'\\'+today):
    os.mkdir(folder+'\\'+today)
folder = folder+'\\'+today

speed =  265
volume=1000
liquid_name = 'Viscosity_std_505'
filename = folder + '/' +'/'+ today + "_" + now[:-3] + '_' +liquid_name+'_'+str(speed).replace('.','_') + ".csv"
if mover.getToolPosition()[0] != balance_deck.wells['A1'].from_top((0,0,-10)):
    mover.safeMoveTo(balance_deck.wells['A1'].from_top((0,0,-10)),descent_speed_fraction=0.25)
time.sleep(5)
balance.zero(wait=5)
balance.clearCache()
balance.toggleRecord(on=True)
time.sleep(15)
liquid.aspirate(volume, speed=speed)
while True:
    data = balance.buffer_df
    data['Mass_smooth']= signal.savgol_filter(data['Mass'],91,1)
    data['Mass_derivative_smooth']=data['Mass_smooth'].diff()
    condition=data['Mass_derivative_smooth'].rolling(30).mean().iloc[-1]
    if condition>-0.05:
        break
print('loop stopped')
time.sleep(10)
mover.setSpeed(50)
mover.moveTo(balance_deck.wells['A1'].from_top((0,0,10)))
liquid.dispense(1000, speed=20)
time.sleep(10)
balance.toggleRecord(on=False)
balance.buffer_df.to_csv(filename)


#%%
liquid.dispense(1000, speed=10)
#balance.toggleRecord(on=False)
#balance.buffer_df.to_csv(r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\Viscous_liquids\Sartorious\Mass_balance_flow_Rate\Calibration\test1.csv')
#balance.buffer_df.plot(x='Time', y='Mass')
# %%
balance.buffer_df['ts'] = balance.buffer_df['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
balance.buffer_df['ts']=balance.buffer_df['ts']-balance.buffer_df['ts'][0]
balance.buffer_df['Mass_change']=balance.buffer_df['Mass']-balance.buffer_df['Mass'][0]
#balance.buffer_df['Mass_smooth']= signal.savgol_filter(balance.buffer_df['Mass'],91,1)
#balance.buffer_df['Mass_derivative_smooth']=balance.buffer_df['Mass_smooth'].diff()


# %%
mover.setSpeed(50)
mover.safeMoveTo(balance_deck.wells['A1'].from_top((0,0,-30)))
