#%%
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import plotly.colors
from plotly.subplots import make_subplots
import re

import plotly.offline as pyo
import plotly.graph_objects as go
pyo.init_notebook_mode()

import plotly.io as pio
import os
pio.templates.default = 'seaborn'
#%%
fig = make_subplots(rows=2, cols=1)

# plot_data = pd.read_csv('Viscosity_std_1275_flow_rate.csv')
plot_data = data_fit
colours= ['#1f77b4','#ff7f0e','#92D050']

fig.add_trace(
    go.Scatter(x=plot_data['ts'], y=plot_data['Mass'],mode='lines',marker_color=colours[0]),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=plot_data['ts'], y=plot_data['Mass_sigmoid'],mode='lines',marker_color=colours[1]),
    row=1,col=1
    )
fig.add_trace(
    go.Scatter(x=plot_data['ts'], y=plot_data['Mass'].diff()/plot_data['ts'].diff(),mode='lines',marker_color=colours[0]),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=plot_data['ts'], y=plot_data['Mass_sigmoid'].diff()/plot_data['ts'].diff(),mode='lines',marker_color=colours[1]),
    row=2, col=1
)   
   
fig.update_xaxes(showline=True, linewidth=2, linecolor='black',ticks='outside',showticklabels=False,mirror=True,row=1,col=1,tickfont= dict(size=14), titlefont=dict(size=18))
fig.update_yaxes(showline=True, linewidth=2, linecolor='black',ticks='outside',mirror=True,title_text= 'Mass<br>[mg]',row=1,col=1,tickfont= dict(size=14), titlefont=dict(size=18))
fig.update_xaxes(showline=True, linewidth=2, linecolor='black',ticks='outside',mirror=True,title_text= 'Time<br>[s]',row=2,col=1,tickfont= dict(size=14), titlefont=dict(size=18))
fig.update_yaxes(showline=True, linewidth=2, linecolor='black',ticks='outside',mirror=True,title_text= 'dMass/dTime<br>[mg/s]',row=2,col=1,tickfont= dict(size=14), titlefont=dict(size=18))

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor = 'rgba(255,255,255,255)',showlegend=False)
fig.show()

# %%
from scipy.optimize import curve_fit
data =  pd.read_csv('Viscosity_std_1275_flow_rate.csv')

def sigmoid(x, L ,x0, k,p,b):
    y = (L-b) / (1 + np.exp(k*(x-x0)))**(1/p) + b
    return y


#using data from balance buffer_df above, calculate time in seconds and mass derivatives
data['ts'] = data['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
data['ts']= data['ts']-data['ts'][0]
data_fit = data.where(data['ts']>10).dropna()
data_fit['Mass']=data_fit['Mass']-data_fit['Mass'].iloc[0]
data_fit['Mass_smooth'] = data_fit['Mass_smooth']-data_fit['Mass_smooth'].iloc[0]

p0 = [abs(min(data_fit['Mass'])), data_fit['ts'].iloc[0],1,1,min(data_fit['Mass'])]

popt, pcov = curve_fit(sigmoid, data_fit['ts'], data_fit['Mass'],p0)

mass_sigmoid = sigmoid(data_fit['ts'],popt[0],popt[1],popt[2],popt[3],popt[4])

data_fit.loc[data_fit.index[0]:,'Mass_sigmoid'] = mass_sigmoid

data_fit.loc[data_fit.index[0]:,'Flow_rate'] = mass_sigmoid = mass_sigmoid.diff()/data_fit.loc[data_fit.index[0]:,'ts'].diff()

flow_rate_max = data_fit['Flow_rate'].min()

flow_rate_98 = data_fit.where(data_fit['Flow_rate']<(0.05*flow_rate_max)).dropna()

time_start, time_final = flow_rate_98.iloc[0].loc['ts'],flow_rate_98.iloc[-1].loc['ts']
# %%
