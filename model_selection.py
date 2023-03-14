#%%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from  sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.linear_model as linear_model
from sklearn.model_selection import cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.model_selection import GridSearchCV

#%%
file_name = r'Std_calibrations/Viscosity_std_398.csv'

df = pd.read_csv(file_name)
features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr','poly', 'SVR', 'SGD','KNR','DTR','KR','PLSR','RFR']
model_dict = {}

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled = df_scaled.sample(frac=1,random_state=42)

df_scaled[features] = scaler.fit_transform(df_scaled[features])

X = df_scaled[features]
y = df_scaled[target]
loo = LeaveOneOut()

for model_name in model_list:
    if model_name == 'lin':
        model = linear_model.LinearRegression()
        scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        model_dict[model_name] = abs(scores['test_score']).mean()

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, 
                                        n_restarts_optimizer=10, 
                                        alpha=0.5, 
                                        normalize_y=True)
        model_dict[model_name] = abs(scores['test_score']).mean()

    
    elif model_name == 'poly':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', linear_model.LinearRegression(fit_intercept=False))])          
        scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        model_dict[model_name] = abs(scores['test_score']).mean()

    elif model_name == 'SVR':
        for i in ['linear','rbf','sigmoid','poly']:
            if i == 'linear':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                grid = dict(C=C,epsilon=epsilon)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)
            elif i == 'rbf':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                grid = dict(C=C,epsilon=epsilon, gamma=gamma)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)
            elif i == 'poly':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma, degree= degree, coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)
            else:
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name+'_'+i] = abs(scores['test_score']).mean()   

    elif model_name == 'SGD':
        for i in ['l2-l1','elasticnet']:
            if i == 'l2-l1':
                model = linear_model.SGDRegressor(random_state=42)
                penalty = ['l2','l1']
                alpha = np.arange(0.0001,0.0011,0.0001)
                learning_rate= ['constant','optimal','invscaling','adaptive']
                power_t = np.arange(0.25,2.5,0.25)
                grid = dict(model=model,penalty=penalty,alpha=alpha, 
                            learning_rate=learning_rate, power_t =power_t)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)
            if i == 'elasticnet':
                model = linear_model.SGDRegressor(random_state=42, penalty=i)
                alpha = np.arange(0.0001,0.0011,0.0001)
                l1_ratio = np.arange(0,1.1,0.1)
                learning_rate= ['constant','optimal','invscaling','adaptive']
                power_t = np.arange(0.25,2.5,0.25)
                grid = dict(model=model,alpha=alpha, l1_ratio= l1_ratio,
                            learning_rate=learning_rate, power_t =power_t)                
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
        
        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean()   

    elif model_name == 'KNR':
        model =  KNeighborsRegressor()
        n_neighbors = range(1,11,1)
        weight= ['uniform','distance']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = range(1,110,10)
        p = range(1,11,1)
        grid = dict(n_neighbors=n_neighbors, weight=weight, algorithm=algorithm
                    leaf_size=leaf_size, p=p)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        search.fit(X,y)
        model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean()  

    elif model_name == 'DTR':
        model =  DecisionTreeRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        splitter = ['best','random']
        grid = dict(criterion=criterion, splitter = splitter)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        search.fit(X,y)
        model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean() 

    elif model_name == 'KR':
        model =  KernelRidge()
        alpha = np.arange(0.1,11,1)
        for i in ['most','polynomial', 'sigmoid']:
            if i == 'most':
                kernel = ['additive_chi2', 'chi2', 'linear', 'poly',  'rbf', 'laplacian','cosine']
                grid = dict (alpha=alpha, kernel=kernel)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)           
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
            elif i ==  'polynomial':
                kernel = 'polynomial'
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef=0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)           
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
            else:
                kernel = 'sigmoid'
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef=0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                search.fit(X,y)           
                model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean() 
    
    elif model_name == 'PLSR':
        model =  PLSRegression()
        n_components = range(1,32,10)
        grid = dict(n_components=n_components)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        search.fit(X,y)
        model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean() 

    elif model_name == 'RFR':
        model =  RandomForestRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        grid = dict(criterion=criterion)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        search.fit(X,y)
        model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean() 

model_dict

#%%




# %%
