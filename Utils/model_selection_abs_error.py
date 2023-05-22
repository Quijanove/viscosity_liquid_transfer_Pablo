#%% Imports
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os


from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut, train_test_split
from  sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.linear_model as linear_model
from sklearn.model_selection import cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn import decomposition

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel

import seaborn as sns
from matplotlib import pyplot as plt


%matplotlib qt

#%% Function definitions


def analyze_scores_list_datasets(grid_search,features,target,dir_name,model_name,i=None):
    """
    analyze_scores_list_datasets
    This function performs the crossvalidation of a model for hyperparameter tuning using grid_search_cv
    for a list of .csv files saved in a directory. It computes the mean MAE of each model across the different 
    files. finally returns a df with the values of the average and std MAE,and hyperparameter values of the model 
    with smalles MAE.
    Args:
    dir_name : Path of a directory containing csv files to be analyzed
    grid_search : sklearn.model_selection.GridSearchCV object with predifined model, parameter grid and score type
    model_name : str defning the name of the model used eg. Support Vector Regression 
    i (optional): str further narrowing the type of model used e.g. Linear in Support Vector Regression Linear
    """
    score_df = pd.DataFrame() #initilize dataframe to save scores of all GridSearchCv results for all csv files

    #iterate through directory
    for file_name in os.listdir(dir_name): 
        if os.path.isfile(dir_name+file_name) == True:
            #initilize dictionary to save scores of all GridSearchCv results for a file
            file_dict = {}
            #load csv as DataFrame and process for ML      
            df = pd.read_csv(dir_name+file_name)
            #Perform GridSearchCV with predefined parameters
            scores = analyze_scores(data=df,grid_search=grid_search,features=features,target=target)
            #Dump GridSearchCV scores for all the models into the dictionary
            file_dict = {'mean_test_score'+file_name[:-4]: scores.cv_results_['mean_test_score'],
                                    'std_test_score'+file_name[:-4]:scores.cv_results_['std_test_score']}
            #Dump GridSearchCV scores for all the models into the DataFrame containing information of all files
            score_df= pd.concat([score_df,pd.DataFrame(file_dict)],axis=1)
    #Compute average for test score and std deviation of test scores for each model across all files        
    score_df['overall_test_score'] = abs(score_df.iloc[:,::2]).mean(axis=1)
    score_df['overall_std_test_score'] = score_df.iloc[:,1::2].mean(axis=1)
    #Select model with minimum average score for all files and dump into a DataFrame that will be returned 
    min_index = score_df['overall_test_score'].idxmin()
    df_score_out = score_df.loc[min_index,['overall_test_score','overall_std_test_score']]
    df_score_out['params'] =grid_search.cv_results_['params'][min_index]
    if i == None:
        df_score_out.name = model_name
    else:
        df_score_out.name = model_name+'_'+ str(i) 
    return df_score_out

def analyze_scores(grid_search,data,features,target):
    """
    analyze_predictions_list_datasets
    This function performs the crossvalidation of a model for hyperparameter tuning using grid_search_cv for a 
    DataFrame, returning a GridSearchCV object.
    grid_search : sklearn.model_selection.GridSearchCV object with predifined model, parameter grid and score type
    data : dataFrame containing features and target valeus to be fitted and predicted
    features: list of column names to be used as  X input
    target: list containing a column name used as Y input 
    """
    #scale features 
    scaler = StandardScaler()
    df_scaled = data.where(data['volume']==1000).dropna(how='all').copy()
    unique_index = df_scaled[features].drop_duplicates().index
    df_scaled = df_scaled.loc[unique_index]

    df_scaled[features] = scaler.fit_transform(df_scaled[features])
    X = df_scaled[features]
    y = df_scaled[target]

    #split train/test sets and train model
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    #Return grid_search object with fitted data 
    return grid_search.fit(X_train,y_train)   


def analyze_predictions_list_datasets(model, features,target,dir_name,model_name,i=None):
    """
    analyze_predictions_list_datasets
    This function is used to train and test a model with predefined hyperparameters for each
    csv file in a directory. It saves a csv for each file containing the experimental and predicted 
    values and also a csv file with the aggregated experimental and predicted values for all files    
    model: sklearn predicotr object with specified hyperparameters
    features: list of column names to be used as  X input
    target: list containing a column name used as Y input 
    dir_name : Path of a directory containing csv files to be analyzed
    model_name : str defning the name of the model used eg. Support Vector Regression 
    i (optional): str further narrowing the type of model used e.g. Linear in Support Vector Regression Linear
    """
    #Initialize a dataframe to save the aggregated predicted and experimental values for all files
    all_predictions_df = pd.DataFrame(columns=['experimental','prediction'])
    
    #iterate through the csv files from the directory provided as an arg
    for file_name in os.listdir(dir_name):
        if os.path.isfile(dir_name+file_name) == True:
            #create df from csv file      
            df = pd.read_csv(dir_name+file_name)

            #Fit and test data 
            prediction_df = analyze_predictions(model=model, data=df,features=features,target=target)
            if i != None:
                prediction_df.to_csv(file_name[:-4]+'_'+model_name+'_'+i+'.csv',index=False)
            else:
                prediction_df.to_csv(file_name[:-4]+'_'+model_name+'.csv',index=False)
            #Append prediction and experimental value for this file into an aggregated dataframe
            all_predictions_df = pd.concat([all_predictions_df,prediction_df], ignore_index= True)
    #Save aggregated dataframe
    if i != None:
        all_predictions_df.to_csv(model_name+'_'+i+'_abs.csv',index=False)
    else:
        all_predictions_df.to_csv(model_name+'_abs.csv',index=False)
    return all_predictions_df



def analyze_predictions(model, data,features,target):
    """
    analyze_predictions_list_datasets
    This function is used to train and test a model with predefined hyperparameters for a givem DataFrame. 
    It perfomrs a train test split and returns a DataFrame containign the predicted and real values.
    model: sklearn predicotr object with specified hyperparameters
    data : dataFrame containing features and target valeus to be fitted and predicted
    features: list of column names to be used as  X input
    target: list containing a column name used as Y input 
    """
    #scale features 
    scaler = StandardScaler()
    df_scaled = data.where(data['volume']==1000).dropna(how='all').copy()
    unique_index = df_scaled[features].drop_duplicates().index
    df_scaled = df_scaled.loc[unique_index]    
    df_scaled[features] = scaler.fit_transform(df_scaled[features])
    X = df_scaled[features]
    y = abs(df_scaled[target])

    #split train/test sets and train model
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)     
    model.fit(X_train, y_train)

    #use model to generate prediction and save prediction and experimental value for this file
    prediction  = model.predict(X_test)
    return pd.DataFrame({'experimental':y_test,'prediction':prediction})    

#%%Import csv files that will be analyzed and plot relationships between error and liquid handling parameters
dir_name = r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Std_calibrations'
features = ['aspiration_rate', 'dispense_rate', 'blow_out_rate','delay_aspirate', 'delay_dispense','delay_blow_out']  
target='%error'

for file_name in os.listdir(dir_name):
    if os.path.isfile(dir_name+'/'+file_name)==True:    
        model_dict = {}
        df = pd.read_csv(dir_name+'/'+file_name)
        plot = sns.pairplot(data=df, x_vars=features, y_vars = target, hue = 'volume', palette = 'muted')
        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(file_name[:-4])
        plot.fig.savefig(r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Model_analysis/Exploratory/'+file_name[:-4]+'.png')

#%%#%%Import csv files that will be analyzed and plot relationships between error and liquid handling parameters
def lin_f(x,m,b):
    y = m*x+b
    return y

def inverse_x_f(x,a ,b):
    y = 1/(x+a) + b
    return y




dir_name = r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Summaries/'
features = ['time','aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='Viscosity 10 s-1'

transfer_parameters_df = pd.read_csv(dir_name+'Transfer_Parameters_Summary.csv')

plot =  sns.pairplot(data=transfer_parameters_df, x_vars=features, y_vars = target, palette = 'muted')

for feature in features[:3]:
    xdata = np.array(transfer_parameters_df[feature])
    ydata = np.array(transfer_parameters_df[target])
    counter = 0
    if 'rate' in feature:
        parameters, covariance = curve_fit(inverse_x_f, xdata, ydata)
        fit_a= parameters[0]
        fit_b = parameters[1]
        fit_y= inverse_x_f(xdata,fit_a ,fit_b)
        plot.axes[0][1+counter].plot(xdata,fit_y)
        counter +=1 
    else:
        parameters, covariance = curve_fit(lin_f, xdata, ydata)
        fit_m= parameters[0]
        fit_b = parameters[1]
        fit_y= lin_f(xdata,fit_m ,fit_b)
        plot.axes[0][0].plot(xdata,fit_y)        

plot.fig.subplots_adjust(top=0.9)
plot.fig.suptitle('Viscosity vs Best transfer parameters')


# %% 
# Following code returns a csv containing the pameters with the smallest MAE that predict error given
# an input of liquid handling parameters for various regression models. 
# The parameters are found using grid search and leave one out  cross validation (looCV). For each model 
# I first obtain the MAE for each set of parameters inputed by GridSearchCV, this calculation is done 
# for each of the viscosisty standards. Then the average MAE across the different viscosity stnadards
# is computed for each set of parameters. Finally for each model I record the set of paramters with he
# smalles MAE into a panda dataframe, which is returned at the end for the code. 

dir_name = r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr','poly', 'SVR', 'SGD','KNR','DTR','KR','PLSR','RFR']
loo = LeaveOneOut()
scaler = StandardScaler()

df_out= pd.DataFrame()

for model_name in model_list:
    if model_name == 'lin':
        model = linear_model.LinearRegression()
        grid ={}
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-12, 1e6)) * Matern(
                        length_scale=1.0, length_scale_bounds=(1e-12, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, normalize_y=True)
        alpha= np.arange(0.1,1.1,0.1)
        n_restarts_optimizer= range(0,10)
        grid = dict(alpha=alpha,n_restarts_optimizer=n_restarts_optimizer)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)

        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)


  
    elif model_name == 'poly':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', linear_model.LinearRegression(fit_intercept=False))])          
        grid ={}
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)

    elif model_name == 'SVR':
        for i in ['linear','rbf','sigmoid','poly']:
            if i == 'linear':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                grid = dict(C=C,epsilon=epsilon)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

            elif i == 'rbf':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                grid = dict(C=C,epsilon=epsilon, gamma=gamma)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

            
            elif i == 'poly':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma, degree= degree, coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

            else:
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)


    elif model_name == 'SGD':
        for i in ['l2-l1','elasticnet']:
            if i == 'l2-l1':
                model = linear_model.SGDRegressor(random_state=42)
                penalty = ['l2','l1']
                alpha = np.arange(0.0001,0.0011,0.0001)
                learning_rate= ['constant','optimal','invscaling','adaptive']
                power_t = np.arange(0.25,2.5,0.25)
                grid = dict(penalty=penalty,alpha=alpha, 
                            learning_rate=learning_rate, power_t =power_t)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

            if i == 'elasticnet':
                model = linear_model.SGDRegressor(random_state=42, penalty=i)
                alpha = np.arange(0.0001,0.0011,0.0001)
                l1_ratio = np.arange(0,1.1,0.1)
                learning_rate= ['constant','optimal','invscaling','adaptive']
                power_t = np.arange(0.25,2.5,0.25)
                grid = dict(alpha=alpha, l1_ratio= l1_ratio,
                            learning_rate=learning_rate, power_t =power_t)                
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

    # elif model_name == 'KNR':
    #     model =  KNeighborsRegressor()
    #     n_neighbors = range(1,6,1)
    #     weights= ['uniform','distance']
    #     algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    #     leaf_size = range(1,110,10)
    #     p = range(1,11,1)
    #     grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
    #                 leaf_size=leaf_size, p=p)
    #     search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
    #     df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
    #     df_out.to_csv('model_parameters_train.csv', index=False)



    elif model_name == 'DTR':
        model =  DecisionTreeRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        splitter = ['best','random']
        grid = dict(criterion=criterion, splitter = splitter)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)


    elif model_name == 'KR':
        model =  KernelRidge()
        alpha = np.arange(0.1,11,1)
        for i in ['most','polynomial', 'sigmoid']:
            if i == 'most':
                kernel = ['additive_chi2', 'chi2', 'linear', 'poly',  'rbf', 'laplacian','cosine']
                grid = dict (alpha=alpha, kernel=kernel)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)
       
            elif i ==  'polynomial':
                kernel =['polynomial']
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

            else:
                kernel = ['sigmoid']
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name,i=i)],axis=1)
                df_out.to_csv('model_parameters_train.csv', index=False)

    
    elif model_name == 'PLSR':
        model =  PLSRegression()
        n_components = range(1,32,10)
        grid = dict(n_components=n_components)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)

    elif model_name == 'RFR':
        model =  RandomForestRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        grid = dict(criterion=criterion)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_datasets(search,features,target,dir_name,model_name)],axis=1)
        df_out.to_csv('model_parameters_train.csv', index=False)

df_out.to_csv('model_parameters_train.csv', index=False)



# %% 
# Following code computes the predictions of error given a set of liquid handling  parameters using each
# a set of models that are implemented with teh  optimized hyperparameters obtained using the code above.

dir_name = r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr','poly', 'SVR', 'SGD','KNR','DTR','KR','PLSR','RFR']
loo = LeaveOneOut()

all_predictions_dict = {}
    
for model_name in model_list:
    if model_name == 'lin':
        model = linear_model.LinearRegression()
        all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, 
                                        n_restarts_optimizer=1, 
                                        alpha=0.2, 
                                        )
        all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)

    
    elif model_name == 'poly':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', linear_model.LinearRegression(fit_intercept=False))])          
        all_predictions_dict[model_name] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)


    elif model_name == 'SVR':
        for i in ['linear','rbf','sigmoid','poly']:
            if i == 'linear':
                model =  SVR(kernel=i,C=11,epsilon=0.1)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)

            elif i == 'rbf':
                model =  SVR(kernel=i, C=21, epsilon=0.1, gamma= 'auto')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)

            elif i == 'poly':
                model =  SVR(kernel=i, C = 21, coef0 = 11, degree = 2, epsilon = 0.1, gamma = 'auto')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)

            else:
                model =  SVR(kernel=i, C=11, coef0= 0, epsilon = 1.0,gamma= 'scale')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)


    elif model_name == 'SGD':
        for i in ['l2-l1','elasticnet']:
            if i == 'l2-l1':
                model = linear_model.SGDRegressor(random_state=42, alpha=0.008,learning_rate='adaptive', penalty= 'l2', power_t = 0.25)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)

            if i == 'elasticnet':
                model = linear_model.SGDRegressor(random_state=42, penalty=i,alpha=0.001,learning_rate='adaptive', power_t = 0.25, l1_ratio =0.3)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)

    elif model_name == 'KNR':
        model =  KNeighborsRegressor(algorithm= 'brute',leaf_size=1, n_neighbors=2,p=2,weights='uniform')
        all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)

    elif model_name == 'DTR':
        model =  DecisionTreeRegressor(random_state=42,criterion='friedman_mse', splitter='best')
        all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)

    elif model_name == 'KR':
        for i in ['most','polynomial', 'sigmoid']:
            if i == 'most':
                model =  KernelRidge(kernel='laplacian',alpha=0.1)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)
                   
            elif i ==  'polynomial':
                model =  KernelRidge(alpha=0.1, coef0=11,degree=2,kernel='polynomial')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)
            else:
                model =  KernelRidge(alpha=1.1, coef0=1, degree=1, kernel='sigmoid')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=i)
    
    # elif model_name == 'PLSR':
    #     model =  PLSRegression(n_components = 1)
    #     all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name,i=None)

    elif model_name == 'RFR':
        model =  RandomForestRegressor(random_state=42,criterion='friedman_mse')
        all_predictions_dict[model_name]=analyze_predictions_list_datasets(model,features,target, dir_name,model_name)

#%% Plot the precited vs experimental errors obtained in code above
dir =r'C:\Users\quijanovelascop\OneDrive - A STAR\Documents\GitHub\viscosity_liquid_transfer_Pablo\Model_analysis\Train_test'

for file in os.listdir(dir):
    if os.path.isfile(dir+r'\\'+file):
        if 'model' not in file:
            df = pd.read_csv(dir+r'\\'+file)
            y_pred = df['prediction']
            y_test = df['experimental']
            fig,axs = plt.subplots()
            axs.scatter(y_test,y_pred)
            min = df.min().min()
            max = df.max().max()
            one_2_one = np.linspace(min,max,1000)
            axs.plot(one_2_one,one_2_one,color='black')
            axs.set_xlabel('Experimental error [%]')
            axs.set_ylabel('Predicted error [%]')
            # fig.suptitle('Model {} performance chart, MAE :  {}'.format(file[:-4],round(float(parameter_sumamry.loc[0,file[:-4]]),2)))
            # fig.savefig(file[:-4]+'_performance.png')


#%% Plot and analyze models trained with data from manual calibration experiments with unseen data  
# obtained from ML driven experiments

liquid= 'Viscosity_std_398'
model_list = ['lin','gpr','SVR_poly','SVR_linear','KR_polynomial']


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense']#, 'blow_out_rate', 'delay_blow_out']  
target='%error'
scaler = StandardScaler()

all_398_ML = pd.read_csv(r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Opentrons_experiments/'+liquid+'/all_measurements.csv')
df_test = all_398_ML.drop_duplicates(subset=features, keep='first')
df_test = df_test.sample(7, random_state=42)
df_train = pd.read_csv( r'C:/Users/quijanovelascop/OneDrive - A STAR/Documents/GitHub/viscosity_liquid_transfer_Pablo/Std_calibrations/'+liquid+'.csv')

for model_name in model_list:

    if model_name == 'lin':
        model = linear_model.LinearRegression()


    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, 
                                        n_restarts_optimizer=0, 
                                        alpha=0.4, 
                                        normalize_y=True)

    elif model_name == 'SVR_poly':
        model =  SVR(kernel='poly', C = 11, coef0 = 11, degree = 1, epsilon = 0.4, gamma = 'scale')

    elif model_name== 'SVR_linear':
        model =  SVR(kernel='linear',C=21,epsilon=0.4) 

    elif model_name == 'KR_polynomial':
        model =  KernelRidge(alpha=4.1, coef0=11,degree=2,kernel='polynomial')
    
    X_train = scaler.fit_transform(df_train[features])
    y_train = abs(df_train[target])
    model.fit(X_train,y_train)

    X_test = scaler.transform(df_test[features])
    y_test = abs(df_test[target])

    y_test_predict = model.predict(X_test)
    y_train_predict = model.predict(X_train)

    MAE = mean_absolute_error(y_test, y_test_predict)
    r2 = model.score(X_test,y_test)


    min = pd.concat([df_train,df_test])[target].abs().min()
    max = pd.concat([df_train,df_test])[target].abs().max()
    one2one = np.linspace(min,max,)

    fig, axs = plt.subplots()  
    axs.scatter(y_test,y_test_predict, label = 'test')
    axs.scatter(y_train, y_train_predict, label = 'train')
    axs.plot (one2one,one2one)
    axs.legend()
    axs.set_xlabel('Experimental error [%]')
    axs.set_ylabel('Predicted error [%]')
    fig.suptitle('Model {} for {} \n MAE :  {}, r2:{}'.format(model_name,liquid,round(MAE,2), round(r2,2)))
    fig.savefig(liquid+'_'+model_name+'_unseen_data_performance.png')


# %%
liquid= 'Viscosity_std_398'
all_398_ML = pd.read_csv('Opentrons_experiments/'+liquid+'/all_measurements.csv')
df_test = all_398_ML.drop_duplicates(subset=features, keep='first')
df_test = df_test.sample(7, random_state=42)
df_train = pd.read_csv('Std_calibrations/'+liquid+'.csv')

df_plot = pd.concat([df_train[features],df_test[features]])
df_train= pd.DataFrame(decomposition.PCA(3).fit_transform(df_plot[features])).iloc[:-7]
df_test = pd.DataFrame(decomposition.PCA(3).fit_transform(df_plot[features])).iloc[-7:]

fig= plt.figure()
ax= fig.add_subplot(projection = '3d')
ax.scatter(df_train[0],df_train[1],df_train[2], label = 'Train set')
ax.scatter(df_test[0],df_test[1],df_test[2], label = 'Test set')

ax.set_xlabel('X')
ax.set_xlim([-20,40],auto= False)
ax.set_ylabel('Y')
ax.set_ylim([-5,15],auto= False)
ax.set_zlabel('Z')
ax.set_zlim([-2,8],auto= False)
ax.legend()

fig.suptitle('{} PCA analysis'.format(liquid))
fig.savefig('Model_analysis/{}_PCA_analysis.png'.format(liquid))

# %%
