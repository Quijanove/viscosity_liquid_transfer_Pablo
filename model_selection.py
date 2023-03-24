#%%
import pandas as pd
import numpy as np
import os
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
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib qt


def analyze_scores_list_database(dir_name,grid_search,model_name,i=None):
    score_df = pd.DataFrame()

    for file_name in os.listdir(dir_name):
        if os.path.isfile(dir_name+file_name) == True:
            file_dict = {}    
            df = pd.read_csv(dir_name+file_name)
            df_scaled = df.copy()
            df_scaled = df_scaled.sample(frac=1,random_state=42)

            df_scaled[features] = scaler.fit_transform(df_scaled[features])

            X = df_scaled[features]
            y = df_scaled[target]
            grid_search.fit(X,y)
            file_dict = {'mean_test_score'+file_name[:-4]: grid_search.cv_results_['mean_test_score'],
                                    'std_test_score'+file_name[:-4]:grid_search.cv_results_['std_test_score']}
            score_df= pd.concat([score_df,pd.DataFrame(file_dict)],axis=1)
    score_df['overall_test_score'] = abs(score_df.iloc[:,::2]).mean(axis=1)
    score_df['overall_std_test_score'] = score_df.iloc[:,1::2].mean(axis=1)
    min_index = score_df['overall_test_score'].idxmin()
    df_score_out = score_df.loc[min_index,['overall_test_score','overall_std_test_score']]
    df_score_out['params'] =grid_search.cv_results_['params'][min_index]
    if i == None:
        df_score_out.name = model_name
    else:
        df_score_out.name = model_name+'_'+ str(i) 
    return df_score_out


def analyze_predictions_list_database(model, dir_name,model_name,i=None):
    all_predictions_df = pd.DataFrame(columns=['experimental','prediction'])
    loo = LeaveOneOut()

    for file_name in os.listdir(dir_name):
        if os.path.isfile(dir_name+file_name) == True:
            file_dict = {}    
            df = pd.read_csv(dir_name+file_name)
            df_scaled = df.copy()
            df_scaled = df_scaled.sample(frac=1,random_state=42)

            df_scaled[features] = scaler.fit_transform(df_scaled[features])

            X = df_scaled[features]
            y = df_scaled[target]

            for train_index, test_index in loo.split(X):
                model.fit(X.iloc[train_index], y.iloc[train_index])
                prediction  = model.predict(X.iloc[test_index])
                prediction_df = pd.DataFrame({'experimental':y.iloc[test_index],'prediction':prediction})
                all_predictions_df = pd.concat([all_predictions_df,prediction_df], ignore_index= True)
            if i != None:
                all_predictions_df.to_csv('Model_analysis/'+model_name+'_'+i+'.csv')
            else:
                all_predictions_df.to_csv('Model_analysis/'+model_name+'.csv')
    return all_predictions_df

#%
#%%
dir_name = r'Std_calibrations/'
features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

for file_name in os.listdir(dir_name):    
    model_dict = {}
    df = pd.read_csv(dir_name+file_name)
    plot = sns.pairplot(data=df, x_vars=features, y_vars = target, hue = 'volume', palette = 'muted')
    plot.fig.subplots_adjust(top=0.9)
    plot.fig.suptitle(file_name[:-4])

# %%
dir_name = r'Std_calibrations/'


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
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                        length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, normalize_y=True)
        alpha= np.arange(0.1,1.1,0.1)
        n_restarts_optimizer= range(0,10)
        grid = dict(alpha=alpha,n_restarts_optimizer=n_restarts_optimizer)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)

        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)


    
    elif model_name == 'poly':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', linear_model.LinearRegression(fit_intercept=False))])          
        grid ={}
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)

    elif model_name == 'SVR':
        for i in ['linear','rbf','sigmoid','poly']:
            if i == 'linear':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                grid = dict(C=C,epsilon=epsilon)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)
            
            elif i == 'rbf':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                grid = dict(C=C,epsilon=epsilon, gamma=gamma)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)
            
            
            elif i == 'poly':
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma, degree= degree, coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)

            else:
                model =  SVR(kernel=i)
                C = np.arange(1,110,10)
                epsilon = np.arange(0.1,1.1,0.1)
                gamma = ['scale','auto']
                coef0 = (0,11,1)
                grid = dict(C=C,epsilon=epsilon, gamma=gamma,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)


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
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)
            
            if i == 'elasticnet':
                model = linear_model.SGDRegressor(random_state=42, penalty=i)
                alpha = np.arange(0.0001,0.0011,0.0001)
                l1_ratio = np.arange(0,1.1,0.1)
                learning_rate= ['constant','optimal','invscaling','adaptive']
                power_t = np.arange(0.25,2.5,0.25)
                grid = dict(alpha=alpha, l1_ratio= l1_ratio,
                            learning_rate=learning_rate, power_t =power_t)                
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)

    elif model_name == 'KNR':
        model =  KNeighborsRegressor()
        n_neighbors = range(1,11,1)
        weights= ['uniform','distance']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = range(1,110,10)
        p = range(1,11,1)
        grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                    leaf_size=leaf_size, p=p)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)

        # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
        # model_dict[model_name] = abs(scores['test_score']).mean()  

    elif model_name == 'DTR':
        model =  DecisionTreeRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        splitter = ['best','random']
        grid = dict(criterion=criterion, splitter = splitter)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)

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
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)
        
            elif i ==  'polynomial':
                kernel =['polynomial']
                degree = range(1,4,1)
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)

            else:
                kernel = ['sigmoid']
                coef0 = (0,11,1)
                grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
                search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name,i=i)],axis=1)

    
    elif model_name == 'PLSR':
        model =  PLSRegression()
        n_components = range(1,32,10)
        grid = dict(n_components=n_components)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)

    elif model_name == 'RFR':
        model =  RandomForestRegressor(random_state=42)
        criterion = ['squared_error','friedman_mse','absolute_error','poisson']
        grid = dict(criterion=criterion)
        search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
        df_out = pd.concat([df_out,analyze_scores_list_database(dir_name,search,model_name)],axis=1)
df_out.to_csv('model_parameters_2.csv')




# %%
parameter_sumamry = pd.read_csv('model_parameters_2.csv')
#%%
dir_name = r'Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr','poly', 'SVR', 'SGD','KNR','DTR','KR','PLSR','RFR']
loo = LeaveOneOut()
scaler = StandardScaler()

all_predictions_dict = {}
    
for model_name in model_list:
    if model_name == 'lin':
        model = linear_model.LinearRegression()
        all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

    elif model_name =='gpr':
        matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                    length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

        model = GaussianProcessRegressor(kernel=matern_tunable, 
                                        n_restarts_optimizer=0, 
                                        alpha=0.4, 
                                        normalize_y=True)
        all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

    
    elif model_name == 'poly':
        model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', linear_model.LinearRegression(fit_intercept=False))])          
        all_predictions_dict[model_name] = analyze_predictions_list_database(model, dir_name,model_name,i=None)


    elif model_name == 'SVR':
        for i in ['linear','rbf','sigmoid','poly']:
            if i == 'linear':
                model =  SVR(kernel=i,C=11,epsilon=0.4)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)

            elif i == 'rbf':
                model =  SVR(kernel=i, C=21, epsilon=0.3, gamma= 'auto')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)

            elif i == 'poly':
                model =  SVR(kernel=i, C = 11, coef0 = 11, degree = 1, epsilon = 0.4, gamma = 'scale')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)

            else:
                model =  SVR(kernel=i, C=11, coef0= 0, epsilon = 0.,gamma= 'auto')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)


    elif model_name == 'SGD':
        for i in ['l2-l1','elasticnet']:
            if i == 'l2-l1':
                model = linear_model.SGDRegressor(random_state=42, alpha=0.0007,learning_rate='invscaling', penalty= 'l1', power_t = 0.25)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)

            if i == 'elasticnet':
                model = linear_model.SGDRegressor(random_state=42, penalty=i,alpha=0.0007,learning_rate='invscaling', power_t = 0.25, l1_ratio =1)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)

    elif model_name == 'KNR':
        model =  KNeighborsRegressor(algorithm= 'ball_tree',leaf_size=1, n_neighbors=3,p=1,weights='uniform')
        all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

    elif model_name == 'DTR':
        model =  DecisionTreeRegressor(random_state=42,criterion='friedman_mse', splitter='best')
        all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

    elif model_name == 'KR':
        for i in ['most','polynomial', 'sigmoid']:
            if i == 'most':
                model =  KernelRidge(kernel='poly',alpha=2.1)
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)
                   
            elif i ==  'polynomial':
                model =  KernelRidge(alpha=6.1, coef0=11,degree=2,kernel='polynomial')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)
            else:
                model =  KernelRidge(alpha=3.1, coef0=1, degree=1, kernel='sigmoid')
                all_predictions_dict[model_name+'_'+i] = analyze_predictions_list_database(model, dir_name,model_name,i=i)
    
    # elif model_name == 'PLSR':
    #     model =  PLSRegression(n_components = 1)
    #     all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

    elif model_name == 'RFR':
        model =  RandomForestRegressor(random_state=42,criterion='friedman_mse')
        all_predictions_dict[model_name]=analyze_predictions_list_database(model, dir_name,model_name,i=None)

#%%
parameter_sumamry = pd.read_csv('model_parameters_2.csv')

for i in all_predictions_dict:
    fig,axs = plt.subplots()
    axs.scatter(all_predictions_dict[i]['experimental'],all_predictions_dict[i]['prediction'])
    min = all_predictions_dict[i]['experimental'].min()
    max = all_predictions_dict[i]['experimental'].max()
    one_2_one = np.linspace(min,max,1000)
    axs.plot(one_2_one,one_2_one,color='black')
    axs.set_xlabel('Experimental error [%]')
    axs.set_ylabel('Predicted error [%]')
    fig.suptitle('Model {} performance chart, MAE :  {}'.format(i,round(float(parameter_sumamry.loc[0,i]),2)))
    fig.savefig('Model_analysis/'+i+'_performance.png')


#%%
dir_name = r'Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

model_list =['lin','gpr','poly', 'SVR', 'SGD','KNR','DTR','KR','PLSR','RFR']
loo = LeaveOneOut()
scaler = StandardScaler()

df_out= pd.DataFrame()

for file_name in os.listdir(dir_name):    
    model_dict = {}
    df = pd.read_csv(dir_name+file_name)
    df_scaled = df.copy()
    df_scaled = df_scaled.sample(frac=1,random_state=42)

    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    X = df_scaled[features]
    y = df_scaled[target]

    
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
            scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
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
                    grid = dict(penalty=penalty,alpha=alpha, 
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
                    grid = dict(alpha=alpha, l1_ratio= l1_ratio,
                                learning_rate=learning_rate, power_t =power_t)                
                    search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                    search.fit(X,y)
                    model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
            
            # scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
            # model_dict[model_name] = abs(scores['test_score']).mean()   

        elif model_name == 'KNR':
            model =  KNeighborsRegressor()
            n_neighbors = range(1,11,1)
            weights= ['uniform','distance']
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            leaf_size = range(1,110,10)
            p = range(1,11,1)
            grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
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
                    kernel =['polynomial']
                    degree = range(1,4,1)
                    coef0 = (0,11,1)
                    grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
                    search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
                    search.fit(X,y)           
                    model_dict[model_name+'_'+i] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    
                else:
                    kernel = ['sigmoid']
                    coef0 = (0,11,1)
                    grid = dict (alpha=alpha, kernel=kernel, degree=degree,coef0=coef0)
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
    model_dict['liquid'] = file_name[:-4]
    df_out = df_out.append(model_dict, ignore_index =True)
#%%

df_out.to_csv('model_parameters.csv')

# %%
df=pd.read_csv('model_parameters.csv',index_col=False)
for column in df.columns[4:-1]:
    df.insert(df.columns.get_loc(column)+1, column+ '_score',df[column].str.extract('(\s-?\d\.\d+)',expand=False).astype('float'))

# %%
matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

model = GaussianProcessRegressor(kernel=matern_tunable, 
                                 n_restarts_optimizer=10, 
                                 alpha=0.5,normalize_y=True) 
model_dict[model_name] = abs(scores['test_score']).mean()

#%%
model_dict = {}
dir_name = r'Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                        length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

model = GaussianProcessRegressor(kernel=matern_tunable, normalize_y=True)

loo = LeaveOneOut()
scaler = StandardScaler()
model_name='gpr'
df_out = pd.DataFrame()
for file_name in os.listdir(dir_name):    
    df = pd.read_csv(dir_name+file_name)
    df_scaled = df.copy()
    df_scaled = df_scaled.sample(frac=1,random_state=42)

    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    X = df_scaled[features]
    y = df_scaled[target]
    
    alpha= np.arange(0.1,1.1,0.1)
    n_restarts_optimizer= range(0,10)
    grid = dict(alpha=alpha,n_restarts_optimizer=n_restarts_optimizer)
    search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
    search.fit(X,y)
    model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_) 
    model_dict['liquid'] = file_name[:-4]
    df_out = df_out.append(model_dict, ignore_index =True)
# model_dict[model_name] = str(search.best_estimator_) + ' ' +str(search.best_score_)                    

# %%
#%%
model_dict = {}
dir_name = r'Std_calibrations/'


features = ['aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense', 'blow_out_rate', 'delay_blow_out']  
target='%error'

matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                        length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

model = GaussianProcessRegressor(kernel=matern_tunable, normalize_y=True)

loo = LeaveOneOut()
scaler = StandardScaler()

for file_name in os.listdir(dir_name):    
    df = pd.read_csv(dir_name+file_name)
    df_scaled = df.copy()
    df_scaled = df_scaled.sample(frac=1,random_state=42)

    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    X = df_scaled[features]
    y = df_scaled[target]
    
    alpha= np.arange(0.1,1.1,0.1)
    n_restarts_optimizer= range(0,10)
    grid = dict(alpha=alpha,n_restarts_optimizer=n_restarts_optimizer)
    search = GridSearchCV(model,grid,scoring= 'neg_mean_absolute_error', cv = loo)
    search.fit(X,y)
    model_dict[model_name] = abs(scores['test_score']).mean()
#%%

matern_tunable = ConstantKernel(1.0, (1e-5, 1e6)) * Matern(
                        length_scale=1.0, length_scale_bounds=(1e-5, 1e6), nu=2.5)

model = GaussianProcessRegressor(kernel=matern_tunable, 
                                            n_restarts_optimizer=10, 
                                            alpha=0.5,
                                            random_state=42, 
                                            normalize_y=True)
model_dict = {}
for file_name in os.listdir(dir_name):    
    
    df = pd.read_csv(dir_name+file_name)
    df_scaled = df.copy()
    df_scaled = df_scaled.sample(frac=1,random_state=42)

    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    X = df_scaled[features]
    y = df_scaled[target]

    scores = cross_validate(model, X, y, scoring='neg_mean_absolute_error',cv=loo, n_jobs=-1)
  model_dict[model_name] = abs(scores['test_score']).mean()



